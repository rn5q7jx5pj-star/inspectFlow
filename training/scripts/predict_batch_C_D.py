from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.models import Padim

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=str,required=True,
                        help="MVTec的test目录，包含good、broken_small等子目录")
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    return parser.parse_args()


def prediction_rows(
    predictions,
    input_dir : Path,
)->list[dict]:
     rows = []
     seen = set()
     for batch in predictions:
          if batch.pred_score is None or batch.pred_label is None:
               raise RuntimeError("预测结果缺少分数或标签，请检查checkpoint的后处理状态")
          for index,image_path in enumerate(batch.image_path):
               relative_path = Path(image_path).resolve().relative_to(input_dir)
               if len(relative_path.parts) != 2:
                    raise ValueError(f"图片不符合test/类别/图片结构：{relative_path}")
               key = relative_path.as_posix()
               if key in seen:
                    raise RuntimeError(f"重复的预测图片：{key}")
               seen.add(key)
               defect_type = relative_path.parts[0]
               rows.append({
                    "relative_path": key,
                    "defect_type": defect_type,
                    "gt_label": int(defect_type != "good"),
                    "pred_score": float(batch.pred_score[index].detach().cpu().item()),
                    "pred_label": int(batch.pred_label[index].detach().cpu().item()),
               })
     return sorted(rows,key=lambda row : row["relative_path"])



def main()->None:
    args = parse_args()
    model_str = args.model
    input_dir = args.input_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()

    if model_str not in ("PatchCore","Padim"):
        raise ModuleNotFoundError(f"不接受该模型的推理{model_str}")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"测试目录不存在或不是目录：{input_dir}")
    if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint不存在或不是文件：{checkpoint}")
    csv_path = output_dir / "predictions0912.csv"
    if csv_path.exists():
        raise FileExistsError(f"结果已存在，请选择新的输出目录：{csv_path}")

    json_path = output_dir / "run_info.json"
    if json_path.exists():
        raise FileExistsError(f"结果已存在，请选择新的输出目录：{json_path}")
    
    dataset = PredictDataset(path=input_dir)
    dataset.image_filenames = sorted(dataset.image_filenames,key=str)
    if len(dataset) == 0:
         raise ValueError(f"目录中没有可预测的图片：{input_dir}")
    for path in dataset.image_filenames:
         if len(Path(path).resolve().relative_to(input_dir).parts) != 2:
             raise ValueError(f"请传入包含类别子目录的test目录：{path}")

    if model_str == "PatchCore":
         model = Patchcore(
              backbone="wide_resnet50_2",
              layers=("layer2","layer3"),
              pre_trained=False,
              num_neighbors=9,
              visualizer=False,
         )
    else:
         model = Padim(
              backbone="resnet18",
              layers=["layer1","layer2","layer3"],
              pre_trained=False,
              n_features=100,
              visualizer=False,
         )
    engine = Engine(
         accelerator="cpu",
         devices=1,
         logger=False,
         default_root_dir=str(output_dir),
    )

    start = time.perf_counter()

    predictions = engine.predict(
         model=model,
         dataset=dataset,
         ckpt_path=checkpoint,
         return_predictions=True,
    )

    end = time.perf_counter()

    predict_seconds = end - start

    if not predictions:
        raise RuntimeError("模型没有返回预测结果")

    image_count = sum(
        len(batch.image_path) for batch in predictions
    )

    #计算吞吐量
    throughput = image_count / predict_seconds

    print(f"预测图片数：{image_count}")
    print(f"预测调用耗时：{predict_seconds:.3f} 秒")
    print(f"预测调用吞吐量：{throughput:.2f} 张/秒")

    rows = prediction_rows(predictions, input_dir)
    if len(rows) != len(dataset):
         raise RuntimeError(f"输入{len(dataset)}张，结果却有{len(rows)}行")
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x",encoding="utf-8",newline="") as file:
         writer = csv.DictWriter(file,fieldnames=[
              "relative_path", "defect_type", "gt_label", "pred_score", "pred_label",
         ])
         writer.writeheader()
         writer.writerows(rows)

    predictions_data = {
         "model_Name" : model_str,
         "checkpoint_path" : str(checkpoint),
         "input_path" : str(input_dir),
         "output_path" : str(output_dir),
         "predict_time(s)" : str(predict_seconds)
    }

    with json_path.open("w",encoding="utf-8") as file:
         json.dump(
              predictions_data,
              file,
              ensure_ascii=False,
              indent=2,
         )
         

    print(f"实际checkpoint：{checkpoint}")
    print(f"预测图片数：{len(rows)}")
    print(f"预测调用总耗时：{predict_seconds:.3f}秒")
    print(f"结果CSV：{csv_path}")

if __name__ == "__main__":
    main()
