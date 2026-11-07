from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore


# 1. 指定待检测图片
image_path = "datasets/MVTecAD/bottle/test/broken_small/000.png"

# 2. 指定训练生成的checkpoint
checkpoint_path = (
    "outputs/patchcore_bottle_baseline/"
    "Patchcore/MVTecAD/bottle/v1/"
    "weights/lightning/model.ckpt"
)

# 3. 创建PatchCore模型结构
model = Patchcore()

# 4. 创建待预测数据集
dataset = PredictDataset(
    path=image_path,
    image_size=(256, 256),
)

# 5. 创建Anomalib推理引擎
engine = Engine(
    accelerator="cpu",
    devices=1,
)

# 6. 加载checkpoint并执行预测
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=checkpoint_path,
    return_predictions=True,
)

# 7. 取出第一批预测结果
result = predictions[0]

print(type(predictions))

# 8. 查看结果
print("图片路径：", result.image_path)
print("异常分数：", result.pred_score)
print("预测标签：", result.pred_label)
print("异常热力图形状：", result.anomaly_map.shape)
print("预测掩码形状：", result.pred_mask.shape)