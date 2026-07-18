from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
PATCH_ROOT = REPO / "vllm_patches"
PATCH_PATH = PATCH_ROOT / "patch_fa4_fp8_kv.py"
INSTALL_PATH = PATCH_ROOT / "install.sh"
SPEC = importlib.util.spec_from_file_location("fa4_fp8_kv_patch", PATCH_PATH)
assert SPEC is not None and SPEC.loader is not None
patch_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch_module)


class FA4FP8KVPatchTests(unittest.TestCase):
    def test_full_installer_keeps_fa4_fp8_patch_opt_in(self) -> None:
        source = INSTALL_PATH.read_text()

        self.assertIn(
            'APPLY_FA4_FP8_KV="${AIMO_VLLM_APPLY_FA4_FP8_KV:-0}"',
            source,
        )
        self.assertNotIn(
            'APPLY_FA4_FP8_KV="${AIMO_VLLM_APPLY_FA4_FP8_KV:-1}"',
            source,
        )

    def test_resolve_vllm_root_preserves_venv_python_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary) / "venv"
            interpreter = venv / "bin" / "python"
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to(sys.executable)

            expected = (Path(temporary) / "site-packages" / "vllm", "0.25.1")
            with mock.patch.object(
                patch_module,
                "_installed_vllm",
                return_value=expected,
            ) as locate:
                result = patch_module.resolve_vllm_root(interpreter)

            self.assertEqual(result, expected)
            locate.assert_called_once_with(interpreter.absolute())

    def test_patch_payloads_target_vllm_package(self) -> None:
        for patch_name in patch_module.PATCH_FILES:
            source = (PATCH_ROOT / patch_name).read_text()
            self.assertIn("diff --git a/vllm/", source)
            self.assertNotIn("diff --git a/flash_attn/", source)
            self.assertNotIn("<<<<<<<", source)
            self.assertNotIn(">>>>>>>", source)

    def test_payloads_include_kernel_and_vllm_routing(self) -> None:
        kernel_patch = (PATCH_ROOT / "fa4_fp8_kv_flash_attn.patch").read_text()
        vllm_patch = (PATCH_ROOT / "fa4_fp8_kv_vllm.patch").read_text()

        self.assertIn("fp8_kv_dequant: bool = False", kernel_patch)
        self.assertIn("self.fp8_kv_dequant = fp8_kv_dequant", kernel_patch)
        self.assertIn("get_flash_attn_version() in (3, 4)", vllm_patch)
        self.assertIn('kv_cache_dtype in ("fp8", "fp8_e4m3")', vllm_patch)
        self.assertIn("and block_size != 128", vllm_patch)
        self.assertIn(
            "FA4 FP8 KV cache on SM90 requires block_size=128",
            vllm_patch,
        )
        self.assertIn("fp8_kv_dequant=fa4_fp8_kv_dequant", vllm_patch)
        self.assertIn("diff --git a/vllm/config/vllm.py", vllm_patch)
        self.assertIn("def validate_fa4_fp8_kv_block_size", vllm_patch)
        self.assertIn(
            "current_platform.is_device_capability_family(90)",
            vllm_patch,
        )

    def test_installer_verifies_early_config_guard(self) -> None:
        config_path = Path("config/vllm.py")

        self.assertIn(config_path, patch_module.TARGET_PATHS)
        self.assertIn(config_path, patch_module.REQUIRED_MARKERS)
        self.assertIn(
            "def validate_fa4_fp8_kv_block_size",
            patch_module.REQUIRED_MARKERS[config_path],
        )

    def test_materializes_symlinked_parent_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "base-config"
            source.mkdir()
            (source / "vllm.py").write_text("value = 'base'\n")

            overlay = root / "overlay"
            overlay.mkdir()
            (overlay / "config").symlink_to(source, target_is_directory=True)

            with mock.patch.object(
                patch_module,
                "TARGET_PATHS",
                (Path("config/vllm.py"),),
            ):
                materialized = patch_module._materialize_symlinked_target_parents(
                    overlay
                )

            self.assertEqual(len(materialized), 1)
            self.assertFalse((overlay / "config").is_symlink())
            self.assertTrue(
                (overlay / f"config{patch_module.SYMLINK_BACKUP_SUFFIX}").is_symlink()
            )
            (overlay / "config/vllm.py").write_text("value = 'overlay'\n")
            self.assertEqual((source / "vllm.py").read_text(), "value = 'base'\n")

    def test_materialized_parent_is_restored_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "base-config"
            source.mkdir()
            (source / "vllm.py").write_text("value = 'base'\n")

            overlay = root / "overlay"
            overlay.mkdir()
            link = overlay / "config"
            link.symlink_to(source, target_is_directory=True)

            with mock.patch.object(
                patch_module,
                "TARGET_PATHS",
                (Path("config/vllm.py"),),
            ):
                materialized = patch_module._materialize_symlinked_target_parents(
                    overlay
                )
                patch_module._restore_materialized_parents(materialized)

            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), source.resolve())

    def test_git_apply_helper_is_idempotence_detectable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "vllm/example.py"
            target.parent.mkdir(parents=True)
            target.write_text("value = 1\n")
            patch = root / "example.patch"
            patch.write_text(
                "diff --git a/vllm/example.py b/vllm/example.py\n"
                "--- a/vllm/example.py\n"
                "+++ b/vllm/example.py\n"
                "@@ -1 +1 @@\n"
                "-value = 1\n"
                "+value = 2\n"
            )

            self.assertTrue(
                patch_module._git_apply_check(root, patch, reverse=False)
            )
            patch_module._apply_patch(root, patch)
            self.assertEqual(target.read_text(), "value = 2\n")
            self.assertTrue(
                patch_module._git_apply_check(root, patch, reverse=True)
            )
            self.assertFalse(
                patch_module._git_apply_check(root, patch, reverse=False)
            )


if __name__ == "__main__":
    unittest.main()
