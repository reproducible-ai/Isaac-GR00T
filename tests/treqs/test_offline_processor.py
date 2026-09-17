"""Exercise the ordinary processor boundary with a real cached fast tokenizer."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from huggingface_hub.errors import LocalEntryNotFoundError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "nvidia/Cosmos-Reason2-2B"
REVISION = "a" * 40


def load_builder(processor=PreTrainedTokenizerFast):
    # Import the actual function without unrelated CUDA/image dependencies.
    path = ROOT / "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_processor"
    )
    namespace = {"Path": Path, "Qwen3VLProcessor": processor}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_processor"]


class OfflineProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.cache = Path(cls.temporary.name)
        cls.snapshot = cls.cache / "models--nvidia--Cosmos-Reason2-2B" / "snapshots" / REVISION
        cls.snapshot.mkdir(parents=True)
        # Transformers 4.57 checks Hub metadata for fast vocabularies >100000.
        tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(100010)}, unk_token="t0"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0").save_pretrained(
            cls.snapshot
        )
        (cls.snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_offline_pinned_repository_loads_real_tokenizer_without_metadata_request(self):
        kwargs = {"revision": REVISION, "cache_dir": str(self.cache), "local_files_only": True}
        original = dict(kwargs)
        with patch(
            "huggingface_hub.model_info",
            side_effect=AssertionError("unexpected Hub metadata request"),
        ) as metadata:
            tokenizer = load_builder()(REPOSITORY, kwargs)
        metadata.assert_not_called()
        self.assertEqual(tokenizer.encode("t1 t2"), [1, 2])
        self.assertEqual(kwargs, original)

    def test_global_offline_mode_also_resolves_snapshot(self):
        with (
            patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            patch(
                "huggingface_hub.model_info",
                side_effect=AssertionError("unexpected Hub metadata request"),
            ),
        ):
            tokenizer = load_builder()(
                REPOSITORY, {"revision": REVISION, "cache_dir": str(self.cache)}
            )
        self.assertEqual(tokenizer.encode("t1 t2"), [1, 2])

    def test_explicit_local_snapshot_passes_through(self):
        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=AssertionError("local path needs no resolution"),
        ):
            tokenizer = load_builder()(str(self.snapshot), {"local_files_only": True})
        self.assertEqual(tokenizer.encode("t1 t2"), [1, 2])

    def test_online_loading_preserves_repository_and_options(self):
        processor = Mock()
        kwargs = {"revision": REVISION, "token": "test-token", "trust_remote_code": True}
        with (
            patch("huggingface_hub.constants.HF_HUB_OFFLINE", False),
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=AssertionError("online path must remain unchanged"),
            ),
        ):
            self.assertIs(
                load_builder(processor)(REPOSITORY, kwargs), processor.from_pretrained.return_value
            )
        processor.from_pretrained.assert_called_once_with(REPOSITORY, **kwargs)

    def test_missing_offline_revision_fails_without_falling_back(self):
        with self.assertRaises((LocalEntryNotFoundError, OSError)):
            load_builder()(
                REPOSITORY,
                {"revision": "b" * 40, "cache_dir": str(self.cache), "local_files_only": True},
            )

    def test_missing_processor_has_actionable_error(self):
        with self.assertRaisesRegex(ImportError, "Qwen3VLProcessor is not available"):
            load_builder(None)(REPOSITORY, {})


if __name__ == "__main__":
    unittest.main()
