"""将训练好的PatchCore checkpoint导出为ONNX，并检查模型能否加载。"""

from pathlib import Path
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练生成的model.ckpt路径",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="导出结果的根目录，请使用新的目录",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    # 1. 检查路径，避免误覆盖已有导出结果。
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint不存在：{checkpoint_path}"
        )

    if output_dir.exists():
        raise FileExistsError(
            f"输出目录已经存在，请选择一个新目录：{output_dir}"
        )

    # 路径检查完成后再导入模型相关依赖。
    import onnx
    import onnxruntime as ort

    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # 2. 从自己训练的checkpoint恢复完整模型。
    # 包含网络参数、Memory Bank及已保存的后处理状态。
    # pre_trained=False避免初始化时额外下载预训练权重；
    # 实际模型权重由checkpoint恢复。
    model = Patchcore.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
        pre_trained=False,
    )

    model.cpu()
    model.eval()

    print(f"checkpoint：{checkpoint_path}")
    print(f"Memory Bank形状：{tuple(model.model.memory_bank.shape)}")

    # 3. 创建导出引擎。
    output_dir.mkdir(parents=True, exist_ok=False)

    engine = Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=str(output_dir),
    )

    # 4. 导出ONNX。
    # 模型已经恢复，因此这里不再传ckpt_path。
    exported_path = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=output_dir,
        model_file_name="patchcore_bottle",
        input_size=(256, 256),
        onnx_kwargs={
            # 首版沿用Anomalib 2.6.0的传统导出方式。
            "dynamo": False,
            "opset_version": 14,

            # 禁用动态维度，固定Batch=1、H=W=256。
            "dynamic_axes": {},
        },
    )

    if exported_path is None:
        raise RuntimeError("导出接口没有返回模型路径")

    onnx_path = Path(exported_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"导出文件不存在：{onnx_path}"
        )

    # 5. 检查ONNX图结构是否合法。
    onnx.checker.check_model(str(onnx_path))
    print("ONNX图结构检查通过")

    # 6. 检查ONNX Runtime能否加载，并查看输入输出。
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    print("\n模型输入：")
    for item in session.get_inputs():
        print(
            f"  name={item.name}, "
            f"shape={item.shape}, "
            f"type={item.type}"
        )

    print("\n模型输出：")
    for item in session.get_outputs():
        print(
            f"  name={item.name}, "
            f"shape={item.shape}, "
            f"type={item.type}"
        )

    print(f"\n导出完成：{onnx_path}")
    print("注意：目前只验证了结构和加载，尚未验证数值一致性。")


if __name__ == "__main__":
    main()



