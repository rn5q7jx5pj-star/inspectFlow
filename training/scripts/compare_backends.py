"""
用于比较onnx模型输出与PyTorch评估模式的输出
输入：
    --input-dir：检测图片的路径
    --checkpoint-dir：用于PyTorch模型的推导
    --onnx-dir：onnx保存路径
打印输出各类比较：
    图片异常分数的绝对误差
    异常热力图的平均绝对误差
    异常热力图的最大绝对误差
    最终标签是否一致
    是否存在NaN或Inf
运行逻辑：
    提取输入
    加载模型
    循环输入推理image列表
    结果放入records
    records写入json文件
"""
import argparse
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--onnx-dir",type=Path,required=True)
    parser.add_argument("--report", type=Path, help="可选JSON报告路径；不覆盖已有文件")
    parser.add_argument("--rtol", type=float, default=1e-3, help="浮点比较的相对容差")
    parser.add_argument("--atol", type=float, default=1e-4, help="浮点比较的绝对容差")

    return parser.parse_args()

def prepare_image(image_path: Path) -> np.ndarray:
    """图片转成RGB、float32、NCHW，数值范围为0～1。"""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (256, 256),
            resample=Image.Resampling.BILINEAR,
        )

        # [H, W, C]，uint8 → float32，0～255 → 0～1。
        array = np.asarray(image, dtype=np.float32) / 255.0

    # [H, W, C] → [C, H, W]。
    array = array.transpose(2, 0, 1)

    # [C, H, W] → [1, C, H, W]。
    array = array[np.newaxis, ...]

    # 保证数组在内存中连续排列。
    return np.ascontiguousarray(array)

def collect_images(path: Path) -> list[Path]:
    """支持单张图片，或递归读取文件夹中的图片，固定排序。"""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if path.is_file():
        paths = [path] if path.suffix.lower() in extensions else []
    elif path.is_dir():
        paths = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
    else:
        raise FileNotFoundError(f"输入图片或目录不存在：{path}")
    if not paths:
        raise ValueError(f"没有找到支持的图片：{path}")
    return paths


def compare_output(
    name: str,
    reference: np.ndarray,
    actual: np.ndarray,
    rtol: float,
    atol: float,
) -> dict:
    """以PyTorch为参考；浮点值用容差，标签和掩码要求完全一致。"""
    result = {
        "torch_shape": list(reference.shape),
        "onnx_shape": list(actual.shape),
        "torch_dtype": str(reference.dtype),
        "onnx_dtype": str(actual.dtype),
        "passed": False,
    }
    # 先检查形状，避免NumPy广播把形状错误掩盖掉。
    if reference.shape != actual.shape:
        result["reason"] = "shape_mismatch"
        return result

    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    result["all_finite"] = finite
    if not finite:
        result["reason"] = "non_finite_output"
        return result

    if name in {"pred_label", "pred_mask"}:
        # 不直接astype(bool)，防止把错误的非零数值都转换成True。
        binary = bool(
            np.isin(reference, [0, 1]).all()
            and np.isin(actual, [0, 1]).all()
        )
        different = int(np.count_nonzero(reference != actual))
        result.update({
            "valid_binary_values": binary,
            "mismatch_count": different,
            "mismatch_fraction": different / reference.size,
            "passed": binary and different == 0,
        })
        if name == "pred_label":
            result["torch_value"] = reference.tolist()
            result["onnx_value"] = actual.tolist()
        return result

    # 使用float64计算误差，减少误差统计本身的舍入影响。
    ref = reference.astype(np.float64)
    out = actual.astype(np.float64)
    error = np.abs(out - ref)
    allowed = atol + rtol * np.abs(ref)
    close = error <= allowed
    result.update({
        "mean_abs_error": float(error.mean()),
        "max_abs_error": float(error.max()),
        "outside_tolerance_count": int(np.count_nonzero(~close)),
        "passed": bool(close.all()),
    })
    if name == "pred_score":
        result["torch_value"] = reference.tolist()
        result["onnx_value"] = actual.tolist()
    return result


def main() -> int:
    args = parse_args()
    image_path = args.input_dir.resolve()
    checkpoint_path = args.checkpoint_dir.resolve()
    onnx_path = args.onnx_dir.resolve()
    report_path = args.report.resolve() if args.report else None

    for value in (args.rtol, args.atol):
        if not math.isfinite(value) or value < 0:
            raise ValueError("rtol和atol必须是有限的非负数")
    for path in (checkpoint_path, onnx_path):
        if not path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{path}")
    if report_path and report_path.exists():
        raise FileExistsError(f"报告已存在，请指定新路径：{report_path}")
    images = collect_images(image_path)

    # 参数校验完成后再导入重量级依赖，--help不加载模型。
    import torch
    import onnxruntime as ort
    from anomalib.models import Patchcore

    # 与export_onnx.py相同：恢复完整模型和后处理状态。
    # weights_only=False仅用于你自己训练、可信的checkpoint。
    model = Patchcore.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu",
        weights_only=False, pre_trained=False,
    )
    model.cpu().eval()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("当前脚本只支持单输入ONNX模型")
    input_info = inputs[0]
    if input_info.type != "tensor(float)" or input_info.shape != [1, 3, 256, 256]:
        raise ValueError(
            f"需要float32 [1,3,256,256]输入，实际："
            f"{input_info.type}, {input_info.shape}"
        )
    names = ("pred_score", "anomaly_map", "pred_label", "pred_mask")
    available = {item.name for item in session.get_outputs()}
    if missing := set(names) - available:
        raise ValueError(f"ONNX缺少输出：{sorted(missing)}")

    records = []
    # enumerate返回迭代对象 从1开始编号
    for index, path in enumerate(images, start=1):
        # 一次预处理、两份数值相同的输入。
        # 不在外部再做ImageNet标准化；完整model.forward已经包含该步骤。
        input_array = prepare_image(path)
        with torch.inference_mode():
            # 调用完整model，而不是跳过前后处理的model.model。
            prediction = model(torch.from_numpy(input_array.copy()))
        torch_outputs = {}
        for name in names:
            value = getattr(prediction, name, None)
            if value is None:
                raise RuntimeError(f"PyTorch模型缺少输出：{name}")
            torch_outputs[name] = value.detach().cpu().numpy()

        values = session.run(list(names), {input_info.name: input_array.copy()})
        onnx_outputs = dict(zip(names, values))
        comparisons = {
            name: compare_output(
                name, torch_outputs[name], onnx_outputs[name], args.rtol, args.atol,
            )
            for name in names
        }
        passed = all(item["passed"] for item in comparisons.values())
        records.append({"image_path": str(path), "passed": passed, "outputs": comparisons})

        print(f"[{index}/{len(images)}] {'PASS' if passed else 'FAIL'} {path.name}")
        for name, item in comparisons.items():
            if "max_abs_error" in item:
                print(
                    f"  {name}: mean_abs={item['mean_abs_error']:.6g}, "
                    f"max_abs={item['max_abs_error']:.6g}, passed={item['passed']}"
                )
            elif "mismatch_count" in item:
                print(f"  {name}: 不一致元素数={item['mismatch_count']}, passed={item['passed']}")
            else:
                print(f"  {name}: {item}")

    passed_count = sum(record["passed"] for record in records)
    report = {
        "checkpoint_path": str(checkpoint_path),
        "onnx_path": str(onnx_path),
        "reference": "PyTorch complete model.forward, eval, CPU",
        "provider": "CPUExecutionProvider",
        "input_protocol": "RGB; Pillow BILINEAR 256x256; float32 NCHW; [0,1]",
        "scope": "相同输入下的后端一致性；不验证原始Engine读图流程的一致性或模型准确率",
        "rtol": args.rtol,
        "atol": args.atol,
        "torch_version": str(torch.__version__),
        "onnxruntime_version": ort.__version__,
        "image_count": len(records),
        "passed_count": passed_count,
        "all_passed": passed_count == len(records),
        "results": records,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"报告：{report_path}")
    print(f"通过：{passed_count}/{len(records)}；标签和掩码要求完全一致。")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
