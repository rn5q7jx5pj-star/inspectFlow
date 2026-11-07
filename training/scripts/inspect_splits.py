from pathlib import Path
import argparse

import yaml
from anomalib.data import MVTecAD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取训练配置，检查并保存数据集划分",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="训练YAML配置文件路径",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="train.txt、val.txt、test.txt的保存目录",
    )

    return parser.parse_args()


def save_paths_to_txt(paths, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for path in sorted(paths):
            file.write(str(path) + "\n")


def main() -> None:
    # 1. 获取命令行参数。
    args = parse_args()

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    # 读取完整训练配置，得到Python字典。
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 当前脚本只处理MVTecAD。
    data_config = config["data"]

    if data_config["class_path"] != "anomalib.data.MVTecAD":
        raise ValueError("当前脚本只支持MVTecAD数据集")

    # 提取数据模块的构造参数。
    data_args = dict(data_config["init_args"])

    # YAML中保存的是枚举名称，例如FROM_TEST。
    # 直接调用Python构造函数时，改成对应的字符串值from_test。
    for key in ("test_split_mode", "val_split_mode"):
        if isinstance(data_args.get(key), str):
            data_args[key] = data_args[key].lower()

    # 使用配置中的参数创建数据模块。表示把字典展开成函数参数
    # 如data_args = {"root": "./datasets/MVTecAD"} 转为MVTecAD(root="./datasets/MVTecAD"}
    datamodule = MVTecAD(**data_args)

    # 3. 根据配置建立训练、验证、测试划分。
    # 根据配置读取图片清单，创建训练集、验证集和测试集，并执行数据划分
    datamodule.setup()

    # 4. 从各数据集的样本表提取图片路径。
    train_paths = set(datamodule.train_data.samples["image_path"])
    val_paths = set(datamodule.val_data.samples["image_path"])
    test_paths = set(datamodule.test_data.samples["image_path"])

    # 5. 检查集合是否重叠。
    assert train_paths.isdisjoint(val_paths), "训练集和验证集存在重叠"
    assert train_paths.isdisjoint(test_paths), "训练集和测试集存在重叠"
    assert val_paths.isdisjoint(test_paths), "验证集和测试集存在重叠"

    # 6. 保存路径清单。
    save_paths_to_txt(train_paths, output_dir / "train.txt")
    save_paths_to_txt(val_paths, output_dir / "val.txt")
    save_paths_to_txt(test_paths, output_dir / "test.txt")

    print(f"训练集：{len(train_paths)}张")
    print(f"验证集：{len(val_paths)}张")
    print(f"测试集：{len(test_paths)}张")
    print(f"清单保存位置：{output_dir}")


if __name__ == "__main__":
    main()