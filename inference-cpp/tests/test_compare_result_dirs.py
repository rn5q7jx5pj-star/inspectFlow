"""使用临时数据测试比较逻辑，不修改真实实验输出。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_result_dirs import compare_dirs, compare_float


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.py = self.root / "python"
        self.cpp = self.root / "cpp"
        self.py.mkdir()
        self.cpp.mkdir()
        model = self.root / "model.onnx"
        # 仅测试文件内容哈希；比较器不会加载ONNX。
        model.write_bytes(b"test model")
        np.zeros((1, 3, 256, 256), dtype="<f4").tofile(self.py / "input.f32")
        common = dict(model_path=str(model), input_shape=[1,3,256,256], input_dtype="float32",
                      pred_score=0.2, pred_label=False, predicted_anomaly_pixels=0)
        self.py_report = dict(common, anomaly_map_shape=[256,256], input_format=dict(
            file="input.f32", shape=[1,3,256,256], dtype="float32", layout="NCHW", channels="RGB", byte_order="little"))
        self.cpp_report = dict(common, input_path=str(self.py / "input.f32"), layout="NCHW", channels="RGB",
                               byte_order="little", anomaly_map_shape=[1,1,256,256], anomaly_map_file="anomaly_map.f32",
                               pred_mask_file="pred_mask.u8", pred_mask_dtype="uint8")
        for directory in (self.py, self.cpp):
            np.zeros((256,256), dtype="<f4").tofile(directory / "anomaly_map.f32")
            np.zeros((256,256), dtype="u1").tofile(directory / "pred_mask.u8")
        self.save_reports()

    def tearDown(self):
        self.temp.cleanup()

    def save_reports(self):
        for directory, report in ((self.py,self.py_report),(self.cpp,self.cpp_report)):
            (directory / "prediction.json").write_text(json.dumps(report), encoding="utf-8")

    def test_matching_2d_and_4d(self):
        self.assertTrue(compare_dirs(self.py, self.cpp)["all_passed"])

    def test_wrong_pair_or_model(self):
        other = self.root / "other.onnx"
        other.write_bytes(b"different model")
        self.cpp_report["model_path"] = str(other)
        self.save_reports()
        self.assertFalse(compare_dirs(self.py,self.cpp)["checks"]["model_file"]["passed"])

    def test_changed_input(self):
        other = self.cpp / "input.f32"
        np.ones((1,3,256,256), dtype="<f4").tofile(other)
        self.cpp_report["input_path"] = str(other)
        self.save_reports()
        self.assertFalse(compare_dirs(self.py,self.cpp)["checks"]["input_file"]["passed"])

    def test_heatmap_error(self):
        array = np.zeros((256,256), dtype="<f4")
        array[0,0] = 0.1
        array.tofile(self.cpp / "anomaly_map.f32")
        check = compare_dirs(self.py,self.cpp)["checks"]["anomaly_map"]
        self.assertEqual(check["outside_tolerance_count"],1)

    def test_mask_error(self):
        array = np.zeros((256,256), dtype="u1")
        array[0,0] = 1
        array.tofile(self.cpp / "pred_mask.u8")
        self.assertEqual(compare_dirs(self.py,self.cpp)["checks"]["pred_mask"]["mismatch_pixels"],1)

    def test_truncated_file(self):
        (self.cpp / "anomaly_map.f32").write_bytes(b"1234")
        with self.assertRaises(ValueError):
            compare_dirs(self.py,self.cpp)

    def test_invalid_label(self):
        self.cpp_report["pred_label"] = "false"
        self.save_reports()
        with self.assertRaises(ValueError):
            compare_dirs(self.py,self.cpp)

    def test_nonfinite_and_shape(self):
        self.assertFalse(compare_float([0], [np.nan], 1e-3, 1e-4)["passed"])
        self.assertFalse(compare_float(np.zeros((2,1)), np.zeros((2,)), 1e-3, 1e-4)["passed"])


if __name__ == "__main__":
    unittest.main()
