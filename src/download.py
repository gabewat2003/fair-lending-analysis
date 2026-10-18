"""Download HMDA LAR records from the FFIEC Data Browser API, one CSV per state."""
import time

import requests

from src import config


def state_url(state: str) -> tuple[str, dict]:
    params = {"states": state, "years": str(config.YEAR), **config.API_FILTERS}
    return config.API_URL, params


def download_state(state: str, retries: int = 4) -> None:
    out = config.RAW_DIR / f"hmda_{config.YEAR}_{state}.csv"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out.name} exists")
        return
    url, params = state_url(state)
    tmp = out.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, params=params, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(out)
            with open(out, "rb") as f:
                rows = sum(1 for _ in f) - 1
            print(f"[ok] {out.name}: {rows:,} rows, {out.stat().st_size / 1e6:.1f} MB  ({r.url})")
            return
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"[retry {attempt}/{retries}] {state}: {e}; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed for {state}")


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for state in config.STATES:
        download_state(state)


if __name__ == "__main__":
    main()
