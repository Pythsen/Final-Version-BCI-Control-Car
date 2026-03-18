import os
import sys
from pathlib import Path


def prepare_torch_dlls() -> None:
    """Best-effort DLL bootstrap for Windows PyInstaller/runtime usage."""
    if os.name != "nt" or sys.version_info < (3, 8):
        return

    candidate_dirs = []

    if getattr(sys, "frozen", False):
        internal_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidate_dirs.extend([internal_dir, internal_dir / "torch" / "lib"])
    else:
        project_root = Path(__file__).resolve().parents[2]
        dist_internal = project_root / "dist" / "main_app" / "_internal"
        candidate_dirs.extend([dist_internal, dist_internal / "torch" / "lib"])

    for dll_dir in candidate_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass
