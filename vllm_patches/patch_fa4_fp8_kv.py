#!/usr/bin/env python3
"""Install the SM90 FA4 FP8-KV kernel and vLLM routing changes."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_VERSION = "0.25.1"
BACKUP_SUFFIX = ".pre_fa4_fp8_kv"

PATCH_FILES = (
    "fa4_fp8_kv_flash_attn.patch",
    "fa4_fp8_kv_vllm.patch",
)

TARGET_PATHS = (
    Path("vllm_flash_attn/cute/flash_fwd_sm90.py"),
    Path("vllm_flash_attn/cute/interface.py"),
    Path("vllm_flash_attn/cute/named_barrier.py"),
    Path("vllm_flash_attn/cute/utils.py"),
    Path("v1/attention/backends/flash_attn.py"),
    Path("vllm_flash_attn/flash_attn_interface.py"),
)

REQUIRED_MARKERS = {
    Path("vllm_flash_attn/cute/flash_fwd_sm90.py"): (
        "fp8_kv_dequant: bool = False",
        "self.fp8_kv_dequant = fp8_kv_dequant",
    ),
    Path("vllm_flash_attn/cute/interface.py"): (
        "fp8_kv_dequant: bool = False",
        "elif fp8_kv_dequant:",
    ),
    Path("v1/attention/backends/flash_attn.py"): (
        "get_flash_attn_version() in (3, 4)",
        "FA4 FP8 KV cache on SM90 requires block_size=128",
        "self.vllm_flash_attn_version != 4",
    ),
    Path("vllm_flash_attn/flash_attn_interface.py"): (
        "fa4_fp8_kv_dequant = (",
        "fp8_kv_dequant=fa4_fp8_kv_dequant",
    ),
}


def _installed_vllm(interpreter: Path) -> tuple[Path, str]:
    script = (
        "import pathlib, vllm; "
        "print(pathlib.Path(vllm.__file__).resolve().parent); "
        "print(vllm.__version__)"
    )
    result = subprocess.run(
        [str(interpreter), "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    if len(lines) != 2:
        raise RuntimeError(f"Could not locate vLLM with {interpreter}: {result.stdout}")
    return Path(lines[0]), lines[1]


def _version_from_root(root: Path) -> str | None:
    version_path = root / "_version.py"
    if not version_path.is_file():
        return None
    match = re.search(
        r"^__version__ = version = ['\"]([^'\"]+)",
        version_path.read_text(),
        re.M,
    )
    return match.group(1) if match else None


def resolve_vllm_root(target: Path) -> tuple[Path, str | None]:
    target = target.expanduser().absolute()
    if (target / "vllm_flash_attn/flash_attn_interface.py").is_file():
        return target, _version_from_root(target)
    if target.is_dir():
        target = target / "bin/python"
    if not target.is_file():
        raise RuntimeError(
            f"Expected a vLLM root, venv, or Python executable: {target}"
        )
    return _installed_vllm(target)


def _git_apply_check(site_packages: Path, patch_path: Path, reverse: bool) -> bool:
    command = [
        "git",
        "apply",
        "--check",
        "--ignore-whitespace",
        "--whitespace=nowarn",
    ]
    if reverse:
        command.append("--reverse")
    command.append(str(patch_path))
    result = subprocess.run(
        command,
        cwd=site_packages,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _apply_patch(site_packages: Path, patch_path: Path) -> None:
    subprocess.run(
        [
            "git",
            "apply",
            "--ignore-whitespace",
            "--whitespace=nowarn",
            str(patch_path),
        ],
        cwd=site_packages,
        check=True,
    )


def _verify_sources(vllm_root: Path) -> None:
    for relative_path in TARGET_PATHS:
        path = vllm_root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing FA4 FP8-KV patch target: {path}")
        source = path.read_text()
        compile(source, str(path), "exec")
        for marker in REQUIRED_MARKERS.get(relative_path, ()):
            if marker not in source:
                raise RuntimeError(f"Missing {marker!r} in {path}")


def install(target: Path) -> None:
    vllm_root, version = resolve_vllm_root(target)
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            f"FA4 FP8-KV patch requires vLLM {SUPPORTED_VERSION}, got {version!r}"
        )

    patch_root = Path(__file__).resolve().parent
    site_packages = vllm_root.parent
    patch_states: list[tuple[Path, str]] = []
    for patch_name in PATCH_FILES:
        patch_path = patch_root / patch_name
        if not patch_path.is_file():
            raise RuntimeError(f"Missing patch payload: {patch_path}")
        if _git_apply_check(site_packages, patch_path, reverse=False):
            state = "pending"
        elif _git_apply_check(site_packages, patch_path, reverse=True):
            state = "installed"
        else:
            raise RuntimeError(
                f"Patch does not match vLLM {version}: {patch_path.name}"
            )
        patch_states.append((patch_path, state))

    originals: dict[Path, tuple[bytes, int]] = {}
    for relative_path in TARGET_PATHS:
        path = vllm_root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing FA4 FP8-KV patch target: {path}")
        originals[path] = (path.read_bytes(), path.stat().st_mode)

    try:
        for patch_path, state in patch_states:
            if state == "installed":
                print(f"[vllm-patch] verified: {patch_path.name}")
                continue
            for path in originals:
                backup = path.with_name(path.name + BACKUP_SUFFIX)
                if not backup.exists():
                    shutil.copy2(path, backup)
            _apply_patch(site_packages, patch_path)
            print(f"[vllm-patch] installed: {patch_path.name}")
        _verify_sources(vllm_root)
    except Exception:
        for path, (content, mode) in originals.items():
            path.write_bytes(content)
            os.chmod(path, mode)
        raise

    print(f"[vllm-patch] FA4 FP8-KV sources verified for vLLM {version}")


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else sys.executable)
    install(target)


if __name__ == "__main__":
    main()
