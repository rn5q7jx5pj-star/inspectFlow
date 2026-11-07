import onnx

model_path = (
    "outputs/onnx/patchcore_bottle_r001_split_v1/"
    "weights/onnx/patchcore_bottle.onnx"
)

model = onnx.load(model_path, load_external_data=False)

for node in model.graph.node:
    # 先只看预处理、后处理相关节点。
    if "pre_processor" in node.name or "post_processor" in node.name:
        print("节点名称：", node.name)
        print("运算类型：", node.op_type)
        print("输入名称：", list(node.input))
        print("输出名称：", list(node.output))
        print()