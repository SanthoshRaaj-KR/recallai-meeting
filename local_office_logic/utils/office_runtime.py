import os
import shutil
import subprocess
from pathlib import Path


def get_office_binary() -> str:
    configured = (os.getenv("LOCAL_OFFICE_OFFICE_BINARY") or "").strip()
    candidate = configured or shutil.which("soffice") or shutil.which("libreoffice")
    if not candidate:
        raise RuntimeError(
            "LibreOffice headless is required for legacy/round-trip office conversion. "
            "Set LOCAL_OFFICE_OFFICE_BINARY to the soffice executable."
        )
    return candidate


def convert_with_office(input_path: Path, output_dir: Path, target_ext: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    office_binary = get_office_binary()
    normalized_ext = target_ext.lstrip(".").lower()

    command = [
        office_binary,
        "--headless",
        "--convert-to",
        normalized_ext,
        "--outdir",
        str(output_dir),
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed for {input_path.name}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    converted = output_dir / f"{input_path.stem}.{normalized_ext}"
    if not converted.exists():
        raise RuntimeError(f"Expected converted file was not produced: {converted}")
    return converted
