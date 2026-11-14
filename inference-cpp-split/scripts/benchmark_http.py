#!/usr/bin/env python3
"""Run a small, reproducible load test against InspectFlow's HTTP detection API.

Example:
    python3 scripts/benchmark_http.py --image /path/to/good.png \
        --image /path/to/defect.png --requests 30 --concurrency 2

Uses only Python's standard library. Does not call Dify or an LLM.
"""

import argparse
import csv
import json
import math
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MAX_IMAGE_BYTES = 5 * 1024 * 1024
FIELDS = (
    "index", "image", "http_status", "latency_ms", "request_id",
    "detection_id", "pred_score", "pred_label", "error",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080",
                        help="C++ service base URL (default: %(default)s)")
    parser.add_argument("--image", action="append", required=True, type=Path,
                        help="Test image; repeat this option for multiple images")
    parser.add_argument("--requests", type=int, default=30,
                        help="Total request count (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent clients (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--csv", type=Path,
                        help="Optionally save per-request records to a CSV file")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path not in ("", "/"):
        parser.error("--base-url must be an http(s) origin, e.g. http://127.0.0.1:8080")
    if args.requests <= 0 or args.concurrency <= 0 or args.timeout <= 0:
        parser.error("--requests, --concurrency, and --timeout must be positive")
    return args

#加载图片
def load_images(paths, parser):
    images = []
    for path in paths:
        data = path.read_bytes()
        images.append((str(path), path.name, data))
    return images

#把图片字节包装成一个符合 multipart/form-data 格式的 HTTP 请求体
#boundary 是每次请求使用的分隔标记。服务器靠它识别一个表单部分从哪里开始、在哪里结束。UUID 让不同请求的标记基本不会重复。
def make_multipart(filename, image_bytes):
    boundary = f"inspectflow-{uuid.uuid4().hex}"
    safe_filename = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{safe_filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("utf-8") #请求头
    body = head + image_bytes + f"\r\n--{boundary}--\r\n".encode("ascii") #把三部分拼起来：文件说明 + 原始图片字节 + 结束标记。结尾分隔线多出的 -- 表示整个表单结束。
    return body, f"multipart/form-data; boundary={boundary}"

#错误
def describe_error(payload, fallback):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return f"{error.get('code', 'UNKNOWN')}: {error.get('message', fallback)}"
        if isinstance(error, str):
            return error
    return fallback

#单个请求的处理过程 返回结果row
def one_request(index, image, url, timeout):
    path, filename, image_bytes = image
    row = dict.fromkeys(FIELDS, "")
    row["index"] = index
    row["image"] = path
    body, content_type = make_multipart(filename, image_bytes)
    request = Request(url, data=body, method="POST",
                      headers={"Content-Type": content_type})

    # Timing starts immediately before network I/O, after the image is loaded.
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            row["http_status"] = response.status
            raw = response.read()
            row["request_id"] = response.headers.get("X-Request-ID", "")
    except HTTPError as exc:
        row["http_status"] = exc.code
        row["request_id"] = exc.headers.get("X-Request-ID", "")
        raw = exc.read()
    except (URLError, TimeoutError, OSError) as exc:
        row["error"] = f"NETWORK_ERROR: {exc}"
        raw = b""
    finally:
        row["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)

    if raw:
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            row["error"] = "INVALID_JSON_RESPONSE"
            return row
        if not isinstance(payload, dict):
            row["error"] = "INVALID_JSON_RESPONSE: expected object"
            return row
        row["request_id"] = payload.get("request_id") or row["request_id"]
        if row["http_status"] == 200:
            for field in ("detection_id", "pred_score", "pred_label"):
                if field in payload:
                    row[field] = payload[field]
            missing = [name for name in ("detection_id", "pred_score", "pred_label")
                       if name not in payload]
            if missing:
                row["error"] = "MISSING_FIELDS: " + ", ".join(missing)
        else:
            row["error"] = describe_error(payload, "HTTP error")
    elif row["http_status"] and not row["error"]:
        row["error"] = "EMPTY_RESPONSE"
    return row

#计算P50和P95
def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main():
    args = parse_args()
    parser = argparse.ArgumentParser()  # Used only for friendly image read errors.
    images = load_images(args.image, parser)
    url = args.base_url.rstrip("/") + "/v1/detections"
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, i + 1, images[i % len(images)], url, args.timeout)
                   for i in range(args.requests)]
        rows = [future.result() for future in as_completed(futures)]
    wall_seconds = time.perf_counter() - started
    rows.sort(key=lambda row: row["index"])

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

    if args.csv:
        try:
            with args.csv.open("w", newline="", encoding="utf-8") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=FIELDS)
                csv_writer.writeheader()
                csv_writer.writerows(rows)
        except OSError as exc:
            print(f"Could not write CSV: {exc}", file=sys.stderr)
            return 2

    succeeded = [row for row in rows if row["http_status"] == 200 and not row["error"]]
    refused = sum(row["http_status"] == 503 for row in rows)
    failed = len(rows) - len(succeeded)
    print(f"\nTotal={len(rows)} success={len(succeeded)} failed={failed} "
          f"refused_503={refused} wall_s={wall_seconds:.3f} "
          f"success_rps={len(succeeded) / wall_seconds:.3f}", file=sys.stderr)
    if succeeded:
        latencies = [row["latency_ms"] for row in succeeded]
        print(f"Success latency (ms): p50={percentile(latencies, .50):.3f} "
              f"p95={percentile(latencies, .95):.3f}", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
