from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import colormaps

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用PatchCore checkpoint对单张图片进行异常检测",
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="待检测图片路径",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="PatchCore的model.ckpt路径",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="JSON和热力图的输出目录",
    )

    return parser.parse_args()


def check_input_paths(
    image_path: Path,
    checkpoint_path: Path,
) -> None:
    """检查输入图片和checkpoint是否存在。"""
    if not image_path.exists():
        raise FileNotFoundError(
            f"待检测图片不存在：{image_path}",
        )

    if not image_path.is_file():
        raise ValueError(
            f"图片路径不是文件：{image_path}",
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint不存在：{checkpoint_path}",
        )

    if not checkpoint_path.is_file():
        raise ValueError(
            f"checkpoint路径不是文件：{checkpoint_path}",
        )


def save_heatmap(
    anomaly_map: np.ndarray,
    output_path: Path,
) -> None:
    """将二维异常分数矩阵保存成彩色热力图。预处理"""
    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())

    # 将异常分数归一化到0～1，方便转换成颜色。
    if maximum > minimum:
        normalized_map = (
            anomaly_map - minimum
        ) / (
            maximum - minimum
        )
    else:
        normalized_map = np.zeros_like(
            anomaly_map,
            dtype=np.float32,
        )

    # 使用jet颜色映射：
    # 数值较低的区域偏蓝，数值较高的区域偏红。
    colored_map = colormaps["jet"](normalized_map)

    # matplotlib返回RGBA，这里只保留RGB三个通道。
    rgb_map = colored_map[:, :, :3]

    # 从0～1转换为0～255的uint8图片。
    rgb_map = (rgb_map * 255).astype(np.uint8)

    Image.fromarray(rgb_map).save(output_path)


def main() -> None:
    # args = parse_args()

    # image_path = args.image.resolve()
    # checkpoint_path = args.checkpoint.resolve()
    # output_dir = args.output_dir.resolve()

    image_path = Path("datasets/MVTecAD/bottle/test/good/000.png")
    checkpoint_path = Path("outputs/padim_bottle_baseline/Padim/MVTecAD/bottle/v0/weights/lightning/model.ckpt")
    output_dir = Path("outputs/week04_predict/good000")
    # 1. 检查图片和checkpoint。
    check_input_paths(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
    )

    # 2. 创建输出目录。
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 3. 创建Padim模型结构。
    # 参数与当前Bottle基线保持一致。
    model = Padim(
        backbone="resnet18",
        layers=["layer1","layer2", "layer3"],
        pre_trained=True,
        n_features=100,
    )

    # 4. 创建单图片预测数据集。
    dataset = PredictDataset(
        path=image_path,
        image_size=(256, 256),
    )

    # 5. 创建Anomalib推理引擎。
    engine = Engine(
        accelerator="cpu",
        devices=1,
    )

    # 6. 加载checkpoint并执行预测。
    start_time = time.perf_counter()

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=checkpoint_path,
        return_predictions=True,
    )

    end_time = time.perf_counter()
    # 测量整个 Engine.predict 调用耗时，包含其内部准备与预测。
    # 不包含调用前创建模型、数据集，以及调用后自行保存结果的时间。
    elapsed_ms = (end_time - start_time) * 1000
    print(f"这次预测调用耗时{elapsed_ms}ms,包含调用内部的准备和预测过程")

    if predictions is None or len(predictions) == 0:
        raise RuntimeError("模型没有返回预测结果")

    # 当前只输入一张图片，因此取第一个Batch。
    result = predictions[0]

    # 7. 从Tensor中提取普通Python数据。
    pred_score = float(
        result.pred_score[0]
        .detach()
        .cpu()
        .item()
    )

    pred_label = bool(
        result.pred_label[0]
        .detach()
        .cpu()
        .item()
    )

    anomaly_map = (
        result.anomaly_map[0]
        .detach()
        .cpu()
        .numpy()
    )

    pred_mask = (
        result.pred_mask[0]
        .detach()
        .cpu()
        .numpy()
    )

    total_predict_ms = (
        end_time - start_time
    ) * 1000

    # 8. 保存异常热力图。
    heatmap_path = output_dir / "anomaly_heatmap.png"

    save_heatmap(
        anomaly_map=anomaly_map,
        output_path=heatmap_path,
    )

    # 9. 构造可以写入JSON的普通Python字典。
    prediction_data = {
        "image_path": str(image_path),
        "checkpoint_path": str(checkpoint_path),
        "model_name": "Padim",
        "backbone": "resnet18",
        "pred_score": pred_score,
        "pred_label": pred_label,
        "result": "anomalous" if pred_label else "normal",
        "anomaly_map_shape": list(anomaly_map.shape),
        "anomaly_map_min": float(anomaly_map.min()),
        "anomaly_map_max": float(anomaly_map.max()),
        "predicted_anomaly_pixels": int(pred_mask.sum()),
        "total_predict_ms": total_predict_ms,
        "heatmap_path": str(heatmap_path),
    }

    # 10. 保存JSON。
    json_path = output_dir / "prediction.json"

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prediction_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # 11. 打印主要结果。
    print("推理完成")
    print(f"输入图片：{image_path}")
    print(f"异常分数：{pred_score:.6f}")
    print(f"预测标签：{pred_label}")
    print(
        "预测结果："
        f"{'异常' if pred_label else '正常'}"
    )
    print(
        f"异常像素数量：{int(pred_mask.sum())}"
    )
    print(
        f"本次调用总耗时：{total_predict_ms:.2f} ms"
    )
    print(f"JSON：{json_path}")
    print(f"异常热力图：{heatmap_path}")


if __name__ == "__main__":
    main()