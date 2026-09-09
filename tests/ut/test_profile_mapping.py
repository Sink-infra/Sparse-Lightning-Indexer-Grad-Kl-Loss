import csv
from pathlib import Path
import tempfile
import unittest

from tests.st.scripts.profile_mock_workloads import read_profile


class ProfileMappingTests(unittest.TestCase):
    def test_runtime_launch_log_maps_kernel_names(self):
        configurations = [{"case_id": str(i), "core_dtype": "float16", "weights_dtype": "float16"}
                          for i in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lines = []
            for i, (key, local_index) in enumerate(((21491679489, 0), (16843009, 0), (21491679489, 1))):
                suffix = "" if i == 1 else "_mix_aiv"
                name = f"SparseLightningIndexerGradKlLoss_example_{key}{suffix}"
                path = root / "profiler/OPPROF_example" / name / str(local_index) / "OpBasicInfo_example.csv"
                path.parent.mkdir(parents=True)
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("Op Name", "Task Duration(us)"))
                    writer.writeheader()
                    writer.writerow({"Op Name": name, "Task Duration(us)": 10 + i})
                lines.extend((f"case={i} network_deterministic=0",
                              f"[INFO] <ProfData> Start profiling on device 0, kernel: {name}"))
            log = root / "profile.log"
            log.write_text("\n".join(lines))
            result = read_profile(root, configurations, log)
            self.assertEqual([r["task_duration_us"] for r in result], [10, 11, 12])
            log.write_text("\n".join(lines[:-1]))
            with self.assertRaisesRegex(RuntimeError, "Missing profiler"):
                read_profile(root, configurations, log)

    def test_scalar_and_vector_algorithms_have_independent_sequences(self):
        configurations = [
            {"case_id": str(index), "core_dtype": "float16", "weights_dtype": "float16",
             "s1": 1, "d_query": dimension, "d_query_index": 1, "d_rope": 1}
            for index, dimension in enumerate((7, 8, 7, 8))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lines = []
            for index, (algorithm, local_index) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                key = 16843009 | (algorithm << 32)
                name = f"SparseLightningIndexerGradKlLoss_example_{key}_mix_aiv"
                path = root / "profiler/OPPROF_example" / name / str(local_index) / "OpBasicInfo_example.csv"
                path.parent.mkdir(parents=True)
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("Op Name", "Task Duration(us)"))
                    writer.writeheader()
                    writer.writerow({"Op Name": name, "Task Duration(us)": index + 10})
                lines.extend((f"case={index} network_deterministic=0",
                              f"[INFO] <ProfData> Start profiling on device 0, kernel: {name}"))
            log = root / "profile.log"
            log.write_text("\n".join(lines))
            rows = read_profile(root, configurations, log)
            self.assertEqual([(r["case_id"], r["task_duration_us"]) for r in rows],
                             [(str(index), index + 10) for index in range(4)])

    def test_mixed_templates_use_separate_launch_sequences(self):
        configurations = [
            {"case_id": name, "core_dtype": dtype, "weights_dtype": dtype}
            for name, dtype in (("half_first", "float16"), ("float_first", "float32"),
                                ("half_second", "float16"), ("float_second", "float32"))]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            lines = []
            for key, local_index, duration in ((16843009, 0, 100.), (0, 0, 200.),
                                               (16843009, 1, 300.), (0, 1, 400.)):
                name = f"SparseLightningIndexerGradKlLoss_example_{key}_mix_aiv"
                path = root / "profiler/OPPROF_example" / name / str(local_index) / "OpBasicInfo_example.csv"
                path.parent.mkdir(parents=True)
                with path.open("w") as stream:
                    writer = csv.DictWriter(stream, fieldnames=("Op Name", "Task Duration(us)"))
                    writer.writeheader()
                    writer.writerow({"Op Name": name, "Task Duration(us)": duration})
                lines.extend((f"case={configurations[len(paths)]['case_id']} network_deterministic=0",
                              f"[INFO] <ProfData> Start profiling on device 0, kernel: {name}"))
                paths.append(path)
            log = root / "profile.log"
            log.write_text("\n".join(lines))
            rows = read_profile(root, configurations, log)
            self.assertEqual([(r["case_id"], r["task_duration_us"]) for r in rows],
                             [("half_first", 100.), ("float_first", 200.),
                              ("half_second", 300.), ("float_second", 400.)])
            duplicate = paths[-1].with_name("OpBasicInfo_duplicate.csv")
            duplicate.write_bytes(paths[-1].read_bytes())
            with self.assertRaisesRegex(RuntimeError, "Missing or duplicate"):
                read_profile(root, configurations, log)
            duplicate.unlink()
            paths[-1].unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing or duplicate"):
                read_profile(root, configurations, log)


if __name__ == "__main__":
    unittest.main()
