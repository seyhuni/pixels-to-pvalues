"""Pack every small result file (csv/json/yaml/md, no model weights / prob-maps) into
results_for_paper.zip - this is what Claude needs for writing the paper.

python scripts/08_pack_results.py
"""
import zipfile

import _common
from frpv.config import REPO_ROOT

KEEP = {".csv", ".json", ".yaml", ".md", ".log", ".txt"}


def main():
    out = REPO_ROOT / "results_for_paper.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for base in [REPO_ROOT / "results", REPO_ROOT / "cache"]:
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in KEEP and "probmaps" not in p.parts:
                    z.write(p, p.relative_to(REPO_ROOT))
                    n += 1
    print(f"{n} files -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
