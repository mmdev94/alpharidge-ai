#!/usr/bin/env python3
"""Poll AlphaRidge (SN45) neuron registration price via TaoMarketCap.

Page:
  https://taomarketcap.com/subnets/45/registration

Live JSON endpoints used by the page:
  GET https://api.taomarketcap.com/internal/v1/subnets/45/
  GET https://api.taomarketcap.com/internal/v1/subnets/45/line-chart/?type=registration

Usage:
  python3 watch_sn45_reg_price.py
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

API_BASE = "https://api.taomarketcap.com"
DEFAULT_NETUID = 45  # AlphaRidge
RAO_PER_TAO = 1_000_000_000


def _get_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "alpharidge-watch-reg-price/2.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_subnet(netuid: int) -> dict:
    row = _get_json(f"{API_BASE}/internal/v1/subnets/{netuid}/")
    if not isinstance(row, dict) or not isinstance(row.get("latest_snapshot"), dict):
        raise RuntimeError(f"no live subnet snapshot for netuid={netuid}: {row!r}")
    return row


def fetch_tao_usd(netuid: int) -> float:
    chart = _get_json(
        f"{API_BASE}/internal/v1/subnets/{netuid}/line-chart/?type=registration"
    )
    if not isinstance(chart, list) or not chart:
        raise RuntimeError(f"no TAO chart price for netuid={netuid}: {chart!r}")
    price = chart[-1].get("tao_price_usd")
    if price is None:
        raise RuntimeError(f"tao_price_usd missing from latest chart point: {chart[-1]!r}")
    return float(price)


def rao_to_tao(rao) -> float:
    return int(rao) / RAO_PER_TAO


def format_line(row: dict, netuid: int, tao_usd: float) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot = row["latest_snapshot"]
    cost_rao = snapshot.get("burn")
    if cost_rao is None:
        raise RuntimeError(f"burn missing from latest_snapshot: keys={list(snapshot)}")
    cost_tao = rao_to_tao(cost_rao)
    cost_usd = cost_tao * tao_usd
    block = snapshot.get("block_number")
    regs = snapshot.get("registrations_this_interval")
    allowed = snapshot.get("network_registration_allowed")
    name = row.get("name") or row.get("subnet_name") or "AlphaRidge"
    return (
        f"{now}  netuid={netuid} ({name})  "
        f"reg_price={cost_tao:.9f} τ  (${cost_usd:,.2f})  "
        f"tao=${tao_usd:,.2f}  "
        f"block={block}  "
        f"regs_this_interval={regs}  registration_allowed={allowed}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Watch SN45 registration price (TaoMarketCap)")
    p.add_argument("--netuid", type=int, default=DEFAULT_NETUID, help="Subnet netuid (default 45)")
    p.add_argument("--interval", type=float, default=60.0, help="Poll interval seconds (default 30)")
    p.add_argument("--once", action="store_true", help="Print once and exit")
    args = p.parse_args()

    print(
        f"Polling TaoMarketCap live snapshot netuid={args.netuid} every {args.interval}s  "
        f"(Ctrl+C to stop)",
        flush=True,
    )

    while True:
        try:
            row = fetch_subnet(args.netuid)
            tao_usd = fetch_tao_usd(args.netuid)
            print(format_line(row, args.netuid, tao_usd), flush=True)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            print(f"{datetime.now(timezone.utc).isoformat()}  ERROR HTTP {e.code}: {detail}", flush=True)
        except Exception as e:
            print(
                f"{datetime.now(timezone.utc).isoformat()}  ERROR {type(e).__name__}: {e}",
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
        raise SystemExit(0)
