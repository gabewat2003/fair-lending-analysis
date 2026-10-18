"""CLI entry point: python -m src.pipeline {data|clean|analyze|all}"""
import sys
import time

from src import clean, disparities, download, proxies, regression, report

STAGES = {
    "data": [download.main],
    "clean": [clean.main],
    "analyze": [disparities.main, regression.main, proxies.main, report.main],
}
STAGES["all"] = STAGES["data"] + STAGES["clean"] + STAGES["analyze"]


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: python -m src.pipeline {{{'|'.join(STAGES)}}}")
    for fn in STAGES[sys.argv[1]]:
        t = time.perf_counter()
        print(f"\n=== {fn.__module__} ===")
        fn()
        print(f"--- {fn.__module__} done in {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
