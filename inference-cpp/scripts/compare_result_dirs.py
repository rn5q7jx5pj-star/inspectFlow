"""比较Python ORT与C++ ORT的单张图片结果目录，不重新执行推理。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-dir", type=Path, required=True)
    parser.add_argument("--cpp-dir", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--report", type=Path, help="可选JSON报告；不覆盖已有文件")
    return parser.parse_args()


def read_report(directory: Path) -> dict:
    with (directory / "prediction.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("prediction.json必须是JSON对象")
    return result


def recorded_path(value: str, directory: Path) -> Path:
    """绝对路径直接使用；相对路径按当前目录/结果目录查找，歧义时报错。"""
    path = Path(value)
    if path.is_absolute():
        candidates = {path.resolve()}
    else:
        candidates = {path.resolve(), (directory / path).resolve()}
    existing = [item for item in candidates if item.is_file()]
    if len(existing) != 1:
        raise ValueError(f"记录路径不存在或有歧义：{value}；请确认路径和运行目录")
    return existing[0]


def local_file(directory: Path, name: str) -> Path:
    """输入协议中的file是结果目录内部文件名。"""
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"结果目录内文件不存在或越界：{name}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_files(reference: Path, actual: Path) -> dict:
    ref_hash, actual_hash = sha256(reference), sha256(actual)
    return {
        "passed": ref_hash == actual_hash,
        "python_path": str(reference), "cpp_path": str(actual),
        "python_sha256": ref_hash, "cpp_sha256": actual_hash,
    }


def shape_list(value) -> list[int]:
    if not isinstance(value, list) or not value or any(type(n) is not int or n <= 0 for n in value):
        raise ValueError(f"无效shape：{value}")
    return value


def spatial_shape(value) -> tuple[int, int]:
    shape = shape_list(value)
    if len(shape) == 2:
        return tuple(shape)
    if len(shape) == 4 and shape[:2] == [1, 1]:
        return tuple(shape[2:])
    raise ValueError(f"只支持[H,W]或[1,1,H,W]热力图：{shape}")


def read_binary(path: Path, dtype: str, shape) -> np.ndarray:
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"文件长度错误：{path}；期望{expected}字节，实际{path.stat().st_size}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def compare_float(reference, actual, rtol: float, atol: float) -> dict:
    # 先检查shape，防止广播掩盖错误；以Python输出为参考计算容差。
    ref, out = np.asarray(reference), np.asarray(actual)
    if ref.shape != out.shape:
        return {"passed": False, "reason": "shape_mismatch"}
    if not np.isfinite(ref).all() or not np.isfinite(out).all():
        return {"passed": False, "reason": "NaN_or_Inf"}
    error = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    outside = int(np.count_nonzero(error > atol + rtol * np.abs(ref.astype(np.float64))))
    return {
        "passed": outside == 0,
        "mean_abs_error": float(error.mean()), "max_abs_error": float(error.max()),
        "outside_tolerance_count": outside,
    }


def binary_label(value) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"pred_label必须是bool或整数0/1：{value}")


def compare_dirs(python_dir: Path, cpp_dir: Path, rtol=1e-3, atol=1e-4) -> dict:
    if any(not math.isfinite(v) or v < 0 for v in (rtol, atol)):
        raise ValueError("rtol和atol必须是有限非负数")
    python_dir, cpp_dir = python_dir.resolve(), cpp_dir.resolve()
    py, cpp = read_report(python_dir), read_report(cpp_dir)
    checks = {}
    checks["model_file"] = compare_files(
        recorded_path(py["model_path"], python_dir),
        recorded_path(cpp["model_path"], cpp_dir),
    )
    fmt = py["input_format"]
    py_input = local_file(python_dir, fmt["file"])
    cpp_input = recorded_path(cpp["input_path"], cpp_dir)
    checks["input_file"] = compare_files(py_input, cpp_input)

    py_shape, cpp_shape = shape_list(py["input_shape"]), shape_list(cpp["input_shape"])
    py_protocol = [py_shape, py["input_dtype"], fmt["layout"], fmt["channels"], fmt["byte_order"]]
    cpp_protocol = [cpp_shape, cpp["input_dtype"], cpp["layout"], cpp["channels"], cpp["byte_order"]]
    supported = [[1, 3, 256, 256], "float32", "NCHW", "RGB", "little"]
    checks["input_protocol"] = {
        "passed": py_protocol == cpp_protocol == supported and fmt["shape"] == py_shape and fmt["dtype"] == "float32",
        "python": py_protocol, "cpp": cpp_protocol,
    }
    # 当前教学协议固定；协议不支持时停止解码，不能猜测字节含义。
    if not checks["input_protocol"]["passed"]:
        raise ValueError("输入协议不一致，或不是float32小端RGB/NCHW [1,3,256,256]")
    for path in (py_input, cpp_input):
        data = read_binary(path, "<f4", py_shape)
        if not np.isfinite(data).all() or np.any((data < 0) | (data > 1)):
            raise ValueError(f"输入不是有限的0～1数值：{path}")

    py_spatial, cpp_spatial = spatial_shape(py["anomaly_map_shape"]), spatial_shape(cpp["anomaly_map_shape"])
    checks["output_protocol"] = {
        "passed": py_spatial == cpp_spatial == (256, 256) and cpp["pred_mask_dtype"] == "uint8",
        "python_recorded_shape": py["anomaly_map_shape"],
        "cpp_recorded_shape": cpp["anomaly_map_shape"],
        "normalization": "仅将[1,1,H,W]规范为[H,W]；不任意squeeze或resize",
        "anomaly_map_storage_dtype": "little-endian float32",
        "mask_storage_dtype": "uint8, values 0/1",
        "dtype_evidence": "依据现有.f32/.u8写入协议和文件长度；裸二进制不自带dtype，JSON标量也不保留Tensor dtype",
    }
    if not checks["output_protocol"]["passed"]:
        raise ValueError("输出shape或掩码存储类型不符合当前协议")
    for record in (py, cpp):
        if type(record["pred_score"]) not in (int, float):
            raise ValueError("pred_score必须是JSON数值")
    checks["pred_score"] = compare_float(py["pred_score"], cpp["pred_score"], rtol, atol)
    py_label, cpp_label = binary_label(py["pred_label"]), binary_label(cpp["pred_label"])
    checks["pred_label"] = {"passed": py_label == cpp_label, "python": py_label, "cpp": cpp_label}

    py_map = read_binary(local_file(python_dir, "anomaly_map.f32"), "<f4", py_spatial)
    cpp_map = read_binary(local_file(cpp_dir, cpp["anomaly_map_file"]), "<f4", cpp_spatial)
    checks["anomaly_map"] = compare_float(py_map, cpp_map, rtol, atol)
    py_mask = read_binary(local_file(python_dir, "pred_mask.u8"), "u1", py_spatial)
    cpp_mask = read_binary(local_file(cpp_dir, cpp["pred_mask_file"]), "u1", cpp_spatial)
    if not np.isin(py_mask, [0, 1]).all() or not np.isin(cpp_mask, [0, 1]).all():
        raise ValueError("掩码含有0/1之外的值，不能直接强转bool")
    mismatch = int(np.count_nonzero(py_mask != cpp_mask))
    checks["pred_mask"] = {
        "passed": mismatch == 0, "mismatch_pixels": mismatch,
        "mismatch_fraction": mismatch / py_mask.size,
    }
    checks["mask_pixel_counts"] = {
        "passed": int(py_mask.sum()) == py["predicted_anomaly_pixels"] and int(cpp_mask.sum()) == cpp["predicted_anomaly_pixels"],
        "python_actual": int(py_mask.sum()), "cpp_actual": int(cpp_mask.sum()),
    }
    return {
        "all_passed": all(item["passed"] for item in checks.values()),
        "python_dir": str(python_dir), "cpp_dir": str(cpp_dir),
        "rtol": rtol, "atol": atol, "checks": checks,
        "limitations": [
            "哈希在本次比较时计算；旧结果未保存推理时哈希，不能证明文件在推理后未改变。",
            "只比较主ONNX文件；若模型使用外部权重文件，需额外核验。",
            "这是相同输入的结果一致性检查，不评价模型准确率或读图预处理一致性。",
        ],
    }


def main() -> int:
    args = parse_args()
    try:
        if args.report and args.report.exists():
            raise FileExistsError(f"报告已存在，拒绝覆盖：{args.report}")
        report = compare_dirs(args.python_dir, args.cpp_dir, args.rtol, args.atol)
        for name, check in report["checks"].items():
            print(f"[{'PASS' if check['passed'] else 'FAIL'}] {name}")
            if "max_abs_error" in check:
                print(f"  mean_abs={check['mean_abs_error']:.9g}, max_abs={check['max_abs_error']:.9g}, 超容差元素={check['outside_tolerance_count']}")
            if "mismatch_pixels" in check:
                print(f"  掩码不一致像素：{check['mismatch_pixels']}")
        print("shape按单图协议对齐：[H,W]与[1,1,H,W]视为同一空间布局。")
        print("dtype按文件协议核验；哈希只代表当前文件，无法回溯运行时状态。")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            print(f"报告：{args.report.resolve()}")
        print("全部通过" if report["all_passed"] else "存在不一致，请查看FAIL项")
        return 0 if report["all_passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"检查失败：{error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
