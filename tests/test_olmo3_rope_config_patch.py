import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO / "sglang_patches/patch_olmo3_rope_config_init.py"
SPEC = importlib.util.spec_from_file_location("patch_olmo3_rope_config_init", PATCH_PATH)
PATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PATCH)


UNPATCHED = """class Olmo3Config:
    def __init__(self, max_position_embeddings=2048, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.vocab_size = 50304
        self.max_position_embeddings = max_position_embeddings
"""


class Olmo3RopeConfigPatchTests(unittest.TestCase):
    def test_patch_moves_context_length_before_super_and_is_idempotent(self):
        patched = PATCH.patch_source(UNPATCHED)
        self.assertLess(
            patched.index(PATCH.ASSIGNMENT),
            patched.index(PATCH.SUPER_CALL),
        )
        self.assertEqual(patched.count(PATCH.ASSIGNMENT), 1)
        self.assertEqual(PATCH.patch_source(patched), patched)

    def test_patch_rejects_unknown_source_shape(self):
        with self.assertRaisesRegex(RuntimeError, "max_position_embeddings"):
            PATCH.patch_source(
                UNPATCHED.replace(PATCH.ASSIGNMENT, "")
            )

    def test_patch_venv_creates_backup_and_clears_bytecode(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)
            target = (
                venv
                / "lib/python3.12/site-packages/sglang/srt"
                / PATCH.RELATIVE_PATH
            )
            target.parent.mkdir(parents=True)
            target.write_text(UNPATCHED)
            pyc = target.parent / "olmo3.cpython-312.pyc"
            pyc.write_bytes(b"stale")

            PATCH.patch_venv(venv)

            self.assertEqual(target.read_text(), PATCH.patch_source(UNPATCHED))
            self.assertTrue(
                target.with_suffix(
                    target.suffix + ".pre_olmo3_rope_config_init"
                ).is_file()
            )
            self.assertFalse(pyc.exists())


if __name__ == "__main__":
    unittest.main()
