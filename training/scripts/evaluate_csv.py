from pathlib import Path
import csv
import argparse

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir",type=Path,required=True)
    parser.add_argument("--count",type=int,required=True)
    return parser.parse_args()


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


def main()->None:
    args = parse_args()
    csv_path = args.csv_dir.resolve()
    count = args.count
    verification(csv_path=csv_path,image_count=count)

if __name__ == "__main__":
    main()

