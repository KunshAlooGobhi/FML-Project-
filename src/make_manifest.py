"""Phase 1 / Step 1: record exactly which raw files we used (the 'manifest' in the report).
A SHA-256 checksum is a fingerprint of a file: if even one byte changes, the fingerprint changes,
so anyone can verify they have the identical data."""
import hashlib, json, datetime
import pandas as pd
from config import RAW_DIR, MANIFEST_DIR, RAW_FILES

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

def main():
    MANIFEST_DIR.mkdir(exist_ok=True)
    out = {}
    for name, meta in RAW_FILES.items():
        p = RAW_DIR / name
        df = pd.read_csv(p, low_memory=False)
        out[name] = {
            **meta,
            "retrieved": datetime.date.fromtimestamp(p.stat().st_mtime).isoformat(),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sha256": sha256(p),
        }
        print(f"{name}: {len(df):,} rows, {len(df.columns)} cols")
    (MANIFEST_DIR / "raw_manifest.json").write_text(json.dumps(out, indent=2))
    print("wrote manifests/raw_manifest.json")

if __name__ == "__main__":
    main()
