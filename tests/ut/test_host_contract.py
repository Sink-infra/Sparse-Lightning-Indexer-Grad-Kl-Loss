"""Exercise the real Host validation/Tiling with CANN context builders."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class HostContractTests(unittest.TestCase):
    def test_real_tiling_context(self):
        toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "/home/developer/Ascend/cann-8.5.2"))
        if not (toolkit / "include/base/context_builder/op_tiling_context_builder.h").is_file():
            self.skipTest("CANN context builder is not installed")
        with tempfile.TemporaryDirectory(prefix="sli-host-ut-") as directory:
            binary = Path(directory) / "host_contract"
            command = ["c++", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DOP_TILING_LIB",
                       "-I" + str(toolkit / "include"), str(Path(__file__).with_name("host_contract.cpp")),
                       "-L" + str(toolkit / "lib64"), "-Wl,-rpath," + str(toolkit / "lib64"),
                       "-Wl,--copy-dt-needed-entries", "-lnnopbase", "-lexe_graph", "-lregister", "-ltiling_api", "-lplatform",
                       "-lrt2_registry", "-lunified_dlog", "-lgraph_base", "-lc_sec", "-lmetadef",
                       "-ldl", "-pthread", "-o", str(binary)]
            subprocess.run(command, check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
