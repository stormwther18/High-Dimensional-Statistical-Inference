"""Independent replication of the A2 lower-bound experiment in week 1."""
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

import a2_core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results"


def main():
    configs = [c for c in a2_core.configurations() if c["family"] == "boundary"]
    assert len(configs) == 10
    assert all(c["reps"] == 150 for c in configs)
    assert {(c["gamma"], c["lo"]) for c in configs} == {
        (gamma, lo) for gamma in (1.0, 1.5) for lo in (-4., -3., -2., -1., 0.)
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "A2_config.json").write_text(json.dumps(configs, ensure_ascii=False, indent=2))
    tasks = [(c, rep) for c in configs for rep in range(c["reps"])]
    rows, weights = [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (rr, ww) in enumerate(pool.map(a2_core.one_task, tasks, chunksize=5), 1):
            rows.extend(rr)
            weights.extend(ww)
            if i % 100 == 0:
                print(f"A2: {i}/{len(tasks)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "A2_raw.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "A2_weights.csv", index=False)
    metadata = {
        "seed": 20261022,
        "reference_seed": 20261007,
        "config_repetitions": len(tasks),
        "independent_data_draws": 2 * 150,
        "method_records": len(rows),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": a2_core.np.__version__,
        "scipy": __import__("scipy").__version__,
        "pandas": pd.__version__,
        "core_sha256": hashlib.sha256((ROOT / "Code" / "a2_core.py").read_bytes()).hexdigest(),
        "elapsed_sec": time.perf_counter() - start,
    }
    (OUT / "A2_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
