import csv
from pathlib import Path

csv_path = Path("outputs/week03_batch/test.csv")
FP_path = Path("outputs/week03_batch/false_positives.csv")
FN_path = Path("outputs/week03_batch/false_negatives.csv")


def write_csv(
    csv_path: str | Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    """把字典列表写入CSV，文件已存在时覆盖。"""
    csv_path = Path(csv_path)

    # 输出目录不存在时自动创建。
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()   # 写入列名。
        writer.writerows(rows) # 每个字典写成一行。

def main():
    TP,FP,FN,TN = 0, 0, 0, 0
    false_positives = []  # 保存误报图片。
    false_negatives = [] # 保存漏检图片。
    with csv_path.open(mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            # CSV读取的字段都是字符串，数值字段需要转换。
            relative_path = row["relative_path"]
            gt_label = int(row["gt_label"])
            pred_label = int(row["pred_label"])

            # 在这里处理当前图片的数据。
            if gt_label not in (0, 1) or pred_label not in (0, 1):
                raise ValueError(
                    f"标签必须是0或1：图片={relative_path}，"
                    f"gt_label={gt_label}，pred_label={pred_label}"
                )
            if gt_label == 1:
                if pred_label == 1:
                    TP = TP + 1
                else:
                    FN += 1
                    false_negatives.append(row)
            else:
                if pred_label == 1:
                    FP += 1
                    false_positives.append(row)
                else:
                    TN += 1

    print(f"TP = {TP} | FN = {FN} | FP = {FP} | TN = {TN}")

    fieldnames = [
    "relative_path",
    "defect_type",
    "gt_label",
    "pred_score",
    "pred_label",
    ]

    write_csv(
        csv_path=FP_path,
        fieldnames=fieldnames,
        rows = false_positives
    )

    write_csv(
        csv_path=FN_path,
        fieldnames=fieldnames,
        rows = false_negatives
    )

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


