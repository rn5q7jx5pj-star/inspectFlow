"""使用当前Bottle基线结构，一次加载checkpoint，预测整个MVTec test目录。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="MVTec的test目录，包含good、broken_small等子目录")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def prediction_rows(predictions, input_dir: Path) -> list[dict]:
    """遍历每个Batch中的每张图片；真实标签只用于评测，不输入模型。"""
    rows = []
    seen = set()
    for batch in predictions:
        if batch.pred_score is None or batch.pred_label is None:
            raise RuntimeError("预测结果缺少分数或标签，请检查checkpoint的后处理状态")
        for index, image_path in enumerate(batch.image_path):
            # 遍历取出下标和图片路径
            relative_path = Path(image_path).resolve().relative_to(input_dir)
            # 三步： Path(image_path) string-Path对象/.resolve()转为绝对路径/
            # relative_to() 提取“从测试目录往下怎么走才能找到这张图”。如果图片不在 input_dir 里面，会抛出 ValueError。

            # 当前约定：test/缺陷类型/图片，good表示正常。
            # 检查目录层级是否符合约定 .parts 将路径拆成各个部分，返回一个元组 ("broken_small", "000.png")
            if len(relative_path.parts) != 2:
                raise ValueError(f"图片不符合test/类别/图片结构：{relative_path}")
            # .as_posix() 将Path转换成使用 / 分隔的字符串：
            key = relative_path.as_posix()
            if key in seen:
                raise RuntimeError(f"重复的预测图片：{key}")
            seen.add(key)
            defect_type = relative_path.parts[0]
            # rows 是列表，里面每个元素都是一个字典。append() 将新字典追加到列表末尾，类似C++的 vector.push_back()。
            rows.append({
                "relative_path": key,
                "defect_type": defect_type,
                "gt_label": int(defect_type != "good"),
                "pred_score": float(batch.pred_score[index].detach().cpu().item()),
                "pred_label": int(batch.pred_label[index].detach().cpu().item()),
            })
    return sorted(rows, key=lambda row: row["relative_path"])


def main() -> None:
    # args = parse_args()
    # input_dir = args.input_dir.resolve()
    # checkpoint = args.checkpoint.resolve()
    # output_dir = args.output_dir.resolve()
    input_dir = Path("datasets/MVTecAD/bottle/test").resolve()
    checkpoint = Path("outputs/patchcore_bottle_baseline/Patchcore/MVTecAD/bottle/v2/weights/lightning/model.ckpt").resolve()
    output_dir = Path("outputs/week03_predict/predict_batch").resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"测试目录不存在或不是目录：{input_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint不存在或不是文件：{checkpoint}")
    csv_path = output_dir / "predictions0909.csv"
    if csv_path.exists():
        raise FileExistsError(f"结果已存在，请选择新的输出目录：{csv_path}")

    # 参数检查后再加载依赖；--help不会启动模型。
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # 1. 一次读取整个目录的图片清单，固定顺序。
    dataset = PredictDataset(path=input_dir)
    dataset.image_filenames = sorted(dataset.image_filenames, key=str)
    if len(dataset) == 0:
        raise ValueError(f"目录中没有可预测的图片：{input_dir}")
    for path in dataset.image_filenames:
        if len(Path(path).resolve().relative_to(input_dir).parts) != 2:
            raise ValueError(f"请传入包含类别子目录的test目录：{path}")

    print("图片数量：", len(dataset))
    # 2. 只创建一次模型。沿用predict_single.py的Bottle基线结构。
    # 权重将由完整checkpoint恢复，不额外下载ImageNet权重。
    # 输入缩放和归一化交给模型自带的pre_processor。
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=False,
        num_neighbors=9,
        visualizer=False,
    )
    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=str(output_dir),
    )

    # 3. predict仅调用一次，内部按Batch遍历全部图片并恢复一次checkpoint。
    # Engine为dataset创建的DataLoader默认batch_size=1。
    start = time.perf_counter()
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=checkpoint,
        return_predictions=True,
    )
    predict_seconds = time.perf_counter() - start
    if not predictions:
        raise RuntimeError("模型没有返回预测结果")

    image_count = sum(
        len(batch.image_path) for batch in predictions
    )
    
    if predict_seconds <= 0:
        raise RuntimeError("预测耗时异常")

    throughput = image_count / predict_seconds

    print(f"预测图片数：{image_count}")
    print(f"预测调用耗时：{predict_seconds:.3f} 秒")
    print(f"预测调用吞吐量：{throughput:.2f} 张/秒")

    # 4. 预测完成后，循环处理返回结果，不在循环中重新加载模型。
    rows = prediction_rows(predictions, input_dir)
    if len(rows) != len(dataset):
        raise RuntimeError(f"输入{len(dataset)}张，结果却有{len(rows)}行")
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "relative_path", "defect_type", "gt_label", "pred_score", "pred_label",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"实际checkpoint：{checkpoint}")
    print(f"Memory Bank形状：{model.model.memory_bank.shape}")
    print(f"预测图片数：{len(rows)}")
    print(f"预测调用总耗时：{predict_seconds:.3f}秒")
    print("计时包含checkpoint恢复、预测准备、读图及预测，不含CSV保存；不是纯前向耗时。")
    print(f"结果CSV：{csv_path}")

    verification(csv_path=csv_path,image_count=image_count)

def verification(
    csv_path : Path,
    image_count : int,
):
    if not csv_path.exists():
        raise FileNotFoundError(f"目录文件不存在：{csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    # 换成你CSV中实际使用的路径列名
    paths = [row["relative_path"] for row in rows]

    print("CSV数据行数：", len(rows))
    print("不重复路径数：", len(set(paths)))

    assert len(rows) == image_count, "CSV行数与预测图片数不一致"
    assert len(set(paths)) == len(paths), "CSV中存在重复图片路径"

    TP = len([row for row in rows if row["gt_label"] == "1" and row["pred_label"] == "1"])
    TN = len([row for row in rows if row["gt_label"] == "0" and row["pred_label"] == "0"])
    FP = len([row for row in rows if row["gt_label"] == "0" and row["pred_label"] == "1"])
    FN = len([row for row in rows if row["gt_label"] == "1" and row["pred_label"] == "0"])

    print(f"TP = {TP} | TN = {TN} | FP = {FP} | FN = {FN}")

    try:
        Precision = TP / (TP + FP)
    except ZeroDivisionError:
        print("没有判断为异常的图片！")
        Precision = 0

    try:
        Recall = TP / (TP + FN)
    except ZeroDivisionError:
        print("没有异常的图片！")
        Recall = 0

    try:
        F1 = 2 * Precision * Recall / (Precision + Recall)
    except ZeroDivisionError as e:
        print("分母为0！")
        F1 = 0.0
    
    print(f"Precision = {Precision} | Recall = {Recall} | F1 = {F1}")



if __name__ == "__main__":
    main()
    # verification(csv_path= Path("outputs/week03_predict/predict_batch/test.csv"),
    #             image_count=4
    # )
