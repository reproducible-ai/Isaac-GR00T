"""Reject unverified or oversized attached storage before input preparation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare_droid_calibration", SOURCE / ".treqs/scripts/prepare_droid_calibration.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CalibrationStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def disk(self, name, size, model="Amazon Elastic Block Store"):
        path = self.root / name
        (path / "device").mkdir(parents=True)
        (path / "device/model").write_text(model + "\n")
        (path / "size").write_text(str(size // 512) + "\n")

    def test_counts_all_ebs_disks_and_excludes_bundled_instance_storage(self):
        self.disk("nvme0n1", 200 * 1024**3)
        self.disk("nvme1n1", 4 * 1024**4, "Amazon EC2 NVMe Instance Storage")
        volumes = prepare.storage_inventory(self.root)
        self.assertEqual(len(volumes), 1)
        self.assertEqual(volumes[0]["sizeBytes"], 200 * 1024**3)

    def test_missing_and_oversized_ebs_fail_before_downloads(self):
        with self.assertRaisesRegex(RuntimeError, "outside"):
            prepare.storage_inventory(self.root)
        self.disk("nvme0n1", 800 * 1024**3)
        self.disk("nvme1n1", 400 * 1024**3)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            prepare.storage_inventory(self.root)
