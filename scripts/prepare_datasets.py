"""One-shot dataset preparation.

Run from the loading-service root:

    python -m scripts.prepare_datasets

What it does:

1. **Downloads** the Brunel OR-Library `br1.txt` … `br10.txt` into `data/br/` (skipped if
   already present).
2. **Parses** them into a single structured JSON (`data/br/br_problems.json`) so the runtime
   doesn't re-tokenise text on every load.
3. **Converts** Wadaboa's pickled product pool to parquet at `data/raw/wadaboa_products.parquet`
   (skipped if already present). To get the pickle, clone https://github.com/Wadaboa/3d-bpp
   and pass its path with `--wadaboa-pkl PATH`.
4. **Validates** that everything is loadable.

Why a script and not lazy-on-import?
- Reproducibility for the thesis ("re-running this script bit-exactly regenerates our data").
- Keeps the package import cheap.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from urllib.request import urlretrieve

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
BR_DIR = DATA / "br"
RAW_DIR = DATA / "raw"

BRUNEL_URL_PATTERN = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/br{i}.txt"
BR_FILES = [f"br{i}.txt" for i in range(1, 11)]


def download_br_files(force: bool = False) -> None:
    BR_DIR.mkdir(parents=True, exist_ok=True)
    for fn in BR_FILES:
        dest = BR_DIR / fn
        if dest.exists() and not force:
            continue
        url = BRUNEL_URL_PATTERN.format(i=fn[2:-4])
        print(f"download {url} -> {dest}")
        urlretrieve(url, dest)


def parse_br_file(path: Path) -> list[dict]:
    """Parse one Brunel BR/thpack file into a list of problem dicts."""
    tokens = path.read_text().split()
    pos = 0
    n_problems = int(tokens[pos]); pos += 1
    problems: list[dict] = []
    for _ in range(n_problems):
        pid = int(tokens[pos]); pos += 1
        seed = int(tokens[pos]); pos += 1
        L_cm = int(tokens[pos]); pos += 1
        W_cm = int(tokens[pos]); pos += 1
        H_cm = int(tokens[pos]); pos += 1
        n_types = int(tokens[pos]); pos += 1
        box_types: list[dict] = []
        for _t in range(n_types):
            type_id = int(tokens[pos]); pos += 1
            l = int(tokens[pos]); pos += 1
            vert_l = int(tokens[pos]); pos += 1
            w = int(tokens[pos]); pos += 1
            vert_w = int(tokens[pos]); pos += 1
            h = int(tokens[pos]); pos += 1
            vert_h = int(tokens[pos]); pos += 1
            qty = int(tokens[pos]); pos += 1
            box_types.append({
                "type_id": type_id,
                "length_cm": l,
                "width_cm": w,
                "height_cm": h,
                "allow_vertical_l": bool(vert_l),
                "allow_vertical_w": bool(vert_w),
                "allow_vertical_h": bool(vert_h),
                "quantity": qty,
            })
        problems.append({
            "problem_id": pid,
            "seed_id": seed,
            "container_cm": [L_cm, W_cm, H_cm],
            "box_types": box_types,
        })
    return problems


def parse_all_br() -> dict:
    out: dict = {"source": "Brunel OR-Library", "problems": []}
    for fn in BR_FILES:
        path = BR_DIR / fn
        if not path.exists():
            print(f"skip (missing) {fn}")
            continue
        problems = parse_br_file(path)
        for p in problems:
            p["source_file"] = fn
            out["problems"].append(p)
    print(f"parsed {len(out['problems'])} problems")
    return out


def convert_wadaboa(pkl_path: Path | None) -> None:
    target = RAW_DIR / "wadaboa_products.parquet"
    if target.exists():
        print(f"skip wadaboa (already at {target})")
        return
    if pkl_path is None:
        print(
            "wadaboa pickle not provided. Clone https://github.com/Wadaboa/3d-bpp and pass "
            "--wadaboa-pkl <path-to-products.pkl> to convert."
        )
        return
    import pandas as pd
    with pkl_path.open("rb") as fh:
        df = pickle.load(fh)
    mask = (
        (df["width"] <= 1500) & (df["depth"] <= 1500) & (df["height"] <= 1500)
        & (df["weight"] >= 1) & (df["weight"] <= 1500)
    )
    clean = df[mask].reset_index(drop=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(target, compression="zstd", index=False)
    print(f"wrote {target} ({target.stat().st_size / 1024 / 1024:.1f} MB, {len(clean):,} rows)")


def validate() -> None:
    br_json = BR_DIR / "br_problems.json"
    assert br_json.exists(), "BR JSON missing"
    data = json.loads(br_json.read_text())
    assert len(data["problems"]) > 0, "BR has zero problems"
    print(f"validate: BR JSON has {len(data['problems'])} problems")

    parquet = RAW_DIR / "wadaboa_products.parquet"
    if parquet.exists():
        import pandas as pd
        df = pd.read_parquet(parquet)
        print(f"validate: Wadaboa parquet has {len(df):,} rows, {len(df.columns)} columns")
    else:
        print("validate: Wadaboa parquet missing (skipped)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare loading-service datasets.")
    ap.add_argument("--force", action="store_true", help="redownload existing files")
    ap.add_argument("--wadaboa-pkl", type=Path, default=None, help="path to products.pkl")
    args = ap.parse_args()

    download_br_files(force=args.force)
    parsed = parse_all_br()
    out = BR_DIR / "br_problems.json"
    out.write_text(json.dumps(parsed))
    print(f"wrote {out}")

    convert_wadaboa(args.wadaboa_pkl)
    validate()


if __name__ == "__main__":
    main()
