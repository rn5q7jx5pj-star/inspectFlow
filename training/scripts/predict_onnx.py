"""使用ONNX Runtime对单张图片进行异常检测。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from matplotlib import colormaps
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def save_heatmap(anomaly_map: np.ndarray, path: Path) -> None:
    """使用固定0～1色标，不进行逐图min-max拉伸。"""
    values = np.clip(anomaly_map, 0.0, 1.0)
    rgba = colormaps["jet"](values)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def main() -> None:
    args = parse_args()

    model_path = args.model.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX模型不存在：{model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")
    # if output_dir.exists():
    #     raise FileExistsError(f"请使用新的输出目录：{output_dir}")

    # 1. 加载模型，并单独记录Session创建时间。
    start = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    session_load_ms = (time.perf_counter() - start) * 1000

    # 2. 检查模型输入协议。
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("本脚本只支持单输入模型")

    input_info = inputs[0]

    if input_info.type != "tensor(float)":
        raise ValueError(f"不支持的输入类型：{input_info.type}")
    if input_info.shape != [1, 3, 256, 256]:
        raise ValueError(f"不支持的输入形状：{input_info.shape}")

    print(
        f"输入：{input_info.name}, "
        f"{input_info.shape}, {input_info.type}"
    )

    # 3. 准备输入。不要重复做ImageNet均值/标准差归一化。
    input_array = prepare_image(image_path)

    # 4. 执行推理。
    start = time.perf_counter()
    output_values = session.run(
        None,  # 获取全部输出。
        {input_info.name: input_array},
    )
    session_run_ms = (time.perf_counter() - start) * 1000

    # 用名称关联输出，避免依赖输出顺序。
    output_names = [item.name for item in session.get_outputs()]
    outputs = dict(zip(output_names, output_values))

    required = {"pred_score", "pred_label", "anomaly_map", "pred_mask"}
    missing = required - outputs.keys()
    if missing:
        raise ValueError(f"模型缺少输出：{sorted(missing)}")

    for name, value in outputs.items():
        print(f"输出：{name}, shape={value.shape}, dtype={value.dtype}")

    # 5. 检查输出并提取单张图片结果。
    if outputs["pred_score"].size != 1:
        raise ValueError("pred_score应当只包含一个分数")
    if outputs["pred_label"].size != 1:
        raise ValueError("pred_label应当只包含一个标签")

    for name in ("anomaly_map", "pred_mask"):
        if outputs[name].shape != (1, 1, 256, 256):
            raise ValueError(f"{name}形状不符合预期：{outputs[name].shape}")

    for name in ("pred_score", "anomaly_map"):
        if not np.isfinite(outputs[name]).all():
            raise ValueError(f"{name}包含NaN或Inf")

    pred_score = float(outputs["pred_score"].item())
    pred_label = bool(outputs["pred_label"].item())

    # [1, 1, H, W] → [H, W]。
    anomaly_map = outputs["anomaly_map"][0, 0]
    pred_mask = outputs["pred_mask"][0, 0].astype(bool)

    # 6. 保存图片和原始数值。
    output_dir.mkdir(parents=True, exist_ok=False)

    save_heatmap(anomaly_map, output_dir / "heatmap.png")
    Image.fromarray(
        pred_mask.astype(np.uint8) * 255
    ).save(output_dir / "mask.png")

    np.save(output_dir / "input.npy", input_array)
    np.save(output_dir / "anomaly_map.npy", anomaly_map)
    # 以下为C++/onnx推理输入所需数据
    # 模型输入：小端float32，按C顺序连续保存。
    np.ascontiguousarray(
        input_array, dtype="<f4"
    ).tofile(output_dir / "input.f32")

    # Python ONNX Runtime输出的异常热力图。
    np.ascontiguousarray(
        anomaly_map, dtype="<f4"
    ).tofile(output_dir / "anomaly_map.f32")

    # 预测掩码，保存为0或1；这里不乘255。
    np.ascontiguousarray(
        pred_mask, dtype=np.uint8
    ).tofile(output_dir / "pred_mask.u8")

    # 7. 保存可读的预测记录。
    report = {
        "model_path": str(model_path),
        "image_path": str(image_path),
        "provider": "CPUExecutionProvider",
        "input_shape": list(input_array.shape),
        "input_dtype": str(input_array.dtype),
        "resize": "Pillow BILINEAR, 256x256",
        "pred_score": pred_score,
        "pred_label": pred_label,
        "result": "anomalous" if pred_label else "normal",
        "anomaly_map_shape": list(anomaly_map.shape),
        "anomaly_map_min": float(anomaly_map.min()),
        "anomaly_map_max": float(anomaly_map.max()),
        "predicted_anomaly_pixels": int(pred_mask.sum()),
        "session_load_ms": session_load_ms,
        "session_run_ms": session_run_ms,
        "timing_note": (
            "session_run_ms为首次session.run调用耗时，"
            "不包含外部图片预处理和结果保存，不是稳定性能基准"
        ),
        "input_format": {
        "file": "input.f32",
        "shape": [1, 3, 256, 256],
        "dtype": "float32",
        "byte_order": "little",
        "layout": "NCHW",
        "channels": "RGB",
        "value_range": [0, 1]
        },
    }

    with (output_dir / "prediction.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\n异常分数：{pred_score:.6f}")
    print(f"判断结果：{report['result']}")
    print(f"Session加载：{session_load_ms:.2f} ms")
    print(f"首次预测调用：{session_run_ms:.2f} ms")
    print(f"结果目录：{output_dir}")


if __name__ == "__main__":
    main()