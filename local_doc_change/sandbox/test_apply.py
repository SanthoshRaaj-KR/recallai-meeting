"""Verify SafeApply write-back against the sandbox docs (on temp copies).

Exercises the exact write path the /execute endpoint uses, for .txt (section
regex), .md (heading regex), and .docx (python-docx) — confirming the file is
correctly modified, a backup is created, and the audit log is appended.
"""
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from pipeline.safe_apply import SafeApply  # noqa: E402

CASES = [
    # (filename, section_heading, new_content, must_contain, must_not_contain)
    ("security_policy.txt", "SECTION 1. DATA RETENTION",
     "Application logs are retained for one full year before automatic deletion. "
     "Customer records must be retained for 7 years to satisfy regulatory requirements. "
     "Backups are kept for 30 days on a rolling basis.",
     "one full year", "90 days before automatic deletion"),
    ("security_policy.txt", "SECTION 2. ACCESS CONTROL",
     "Employee access to production systems requires dual approval from the team "
     "manager and the security team. Access is reviewed quarterly. Service accounts "
     "must use rotating credentials that expire every 90 days.",
     "dual approval", "requires approval from a single team manager"),
    ("engineering_handbook.md", "On-Call Rotation",
     "The on-call rotation runs on a half-week basis, with each engineer taking a "
     "half-week of primary on-call duty.",
     "half-week", "full week of primary on-call duty"),
    ("hr_policies.docx", "Paid Time Off",
     "Full-time employees accrue 20 days of paid time off per year.",
     "20 days", "15 days"),
]


def read_text_any(path: Path) -> str:
    if path.suffix == ".docx":
        import docx
        return " ".join(p.text for p in docx.Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sandbox_apply_"))
    shutil.copytree(HERE / "docs", tmp / "docs")
    audit_dir = tmp / "audit"
    applier = SafeApply(audit_dir=str(audit_dir))

    print(f"Working copy: {tmp}\n")
    all_ok = True
    for fname, heading, new_content, must_have, must_not in CASES:
        target = tmp / "docs" / fname
        try:
            backup = applier.apply(
                file_path=str(target),
                section_heading=heading,
                new_content=new_content,
                session_id="apply-test",
                proposal_id=f"p-{fname}-{heading[:10]}",
            )
            text = read_text_any(target)
            backups = list((tmp / "docs").glob(f"{target.stem}.backup.*{target.suffix}"))
            ok_have = must_have in text
            ok_not = must_not not in text
            ok_backup = len(backups) >= 1
            ok = ok_have and ok_not and ok_backup
            all_ok = all_ok and ok
            mark = "OK " if ok else "FAIL"
            print(f"[{mark}] {fname} / {heading}")
            print(f"        has '{must_have}': {ok_have} | removed old: {ok_not} | backup: {ok_backup}")
            if not ok:
                print(f"        --- resulting text (excerpt) ---\n        {text[:300]!r}")
        except Exception as exc:
            all_ok = False
            print(f"[FAIL] {fname} / {heading} -> EXCEPTION: {exc}")

    audit_files = list(audit_dir.glob("*.json")) if audit_dir.exists() else []
    print(f"\nAudit files: {[f.name for f in audit_files]}")
    if audit_files:
        import json
        entries = json.loads(audit_files[0].read_text(encoding="utf-8"))
        print(f"Audit entries: {len(entries)} (expected 4)")

    print("\n" + ("ALL WRITE-BACK CASES PASSED" if all_ok else "SOME CASES FAILED"))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
