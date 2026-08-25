#!/usr/bin/env python3
"""Watch SN45 miner hotkeys; auto burned-register + pm2 restart on deregistration.

Polls the chain every 10 minutes. Checks all miners; if any are deregistered,
re-registers **at most one** per cycle, then ``pm2 restart`` that miner.

Usage (from repo root)::

  pm2 start .venv/bin/python --name alpha-reg-watch -- \\
    watch_sn45_auto_register.py --network finney

  .venv/bin/python watch_sn45_auto_register.py --once --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_NETUID = 45
DEFAULT_NETWORK = "finney"
DEFAULT_INTERVAL_S = 600.0  # 10 minutes
DEFAULT_COOLDOWN_S = 600.0  # skip same miner for 10m after a failed register
RAO_PER_TAO = 1_000_000_000

# Both coldkeys use this password.
WALLET_PASSWORD = "12341234"

# (pm2_name, wallet_name, hotkey_name)
DEFAULT_MINERS: list[tuple[str, str, str]] = [
    ("alpha-miner1", "desearch", "de-miner1"),
    ("alpha-miner2", "alpha-ridge", "alpha-ridge-miner-1"),
    ("alpha-miner3", "alpha-ridge", "alpha-ridge-miner-2"),
    ("alpha-miner4", "alpha-ridge", "alpha-ridge-miner-3"),
    ("alpha-miner5", "alpha-ridge", "alpha-ridge-miner-4"),
    ("alpha-miner6", "alpha-ridge", "alpha-ridge-miner-5"),
    ("alpha-miner7", "alpha-ridge", "alpha-ridge-miner-6"),
    ("alpha-miner8", "alpha-ridge", "alpha-ridge-miner-7"),
    ("alpha-miner9", "alpha-ridge", "alpha-ridge-miner-8"),
    ("alpha-miner10", "alpha-ridge", "alpha-ridge-miner-9"),
    ("alpha-miner11", "alpha-ridge", "alpha-ridge-miner-10"),
    ("alpha-miner12", "alpha-ridge", "alpha-ridge-miner-11"),
    ("alpha-miner13", "alpha-ridge", "alpha-ridge-miner-12"),
    ("alpha-miner14", "alpha-ridge", "alpha-ridge-miner-13"),
    ("alpha-miner15", "alpha-ridge", "alpha-ridge-miner-14"),
    ("alpha-miner16", "alpha-ridge", "alpha-ridge-miner-15"),
    ("alpha-miner17", "alpha-ridge", "alpha-ridge-miner-16"),
    ("alpha-miner18", "alpha-ridge", "alpha-ridge-miner-17"),
    ("alpha-miner19", "alpha-ridge", "alpha-ridge-miner-18"),
    ("alpha-miner20", "alpha-ridge", "alpha-ridge-miner-19"),
    ("alpha-miner21", "alpha-ridge", "alpha-ridge-miner-20"),
    ("alpha-miner22", "alpha-ridge", "alpha-ridge-miner-21"),
    ("alpha-miner23", "alpha-ridge", "alpha-ridge-miner-22"),
    ("alpha-miner24", "alpha-ridge", "alpha-ridge-miner-23"),
    ("alpha-miner25", "alpha-ridge", "alpha-ridge-miner-24"),
    ("alpha-miner26", "alpha-ridge", "alpha-ridge-miner-25"),
    ("alpha-miner27", "alpha-ridge", "alpha-ridge-miner-26"),
    ("alpha-miner28", "alpha-ridge", "alpha-ridge-miner-27"),
    ("alpha-miner29", "alpha-ridge", "alpha-ridge-miner-28"),
    ("alpha-miner30", "alpha-ridge", "alpha-ridge-miner-29"),
    ("alpha-miner31", "alpha-ridge", "alpha-ridge-miner-30"),
    ("alpha-miner32", "alpha-ridge", "alpha-ridge-miner-31"),
    ("alpha-miner33", "alpha-ridge", "alpha-ridge-miner-32"),
    ("alpha-miner34", "alpha-ridge", "alpha-ridge-miner-33"),
    ("alpha-miner35", "alpha-ridge", "alpha-ridge-miner-34"),
    ("alpha-miner36", "alpha-ridge", "alpha-ridge-miner-35"),
    ("alpha-miner37", "alpha-ridge", "alpha-ridge-miner-36"),
    ("alpha-miner38", "alpha-ridge", "alpha-ridge-miner-37"),
    ("alpha-miner39", "alpha-ridge", "alpha-ridge-miner-38"),
    ("alpha-miner40", "alpha-ridge", "alpha-ridge-miner-39"),
    ("alpha-miner41", "alpha-ridge", "alpha-ridge-miner-40"),
    ("alpha-miner42", "alpha-ridge", "alpha-ridge-miner-41"),
    ("alpha-miner43", "alpha-ridge", "alpha-ridge-miner-42"),
]


@dataclass(frozen=True)
class MinerSpec:
    pm2: str
    wallet: str
    hotkey: str


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"{_now()}  {msg}", flush=True)


def load_miners(config_path: Optional[str]) -> list[MinerSpec]:
    if not config_path:
        return [MinerSpec(pm2=p, wallet=w, hotkey=h) for p, w, h in DEFAULT_MINERS]
    path = Path(config_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("miners", data)
    out: list[MinerSpec] = []
    for row in rows:
        out.append(
            MinerSpec(
                pm2=str(row["pm2"]),
                wallet=str(row["wallet"]),
                hotkey=str(row["hotkey"]),
            )
        )
    return out


def make_wallet(bt: Any, name: str, hotkey: str, path: Optional[str]) -> Any:
    kwargs: dict[str, Any] = {"name": name, "hotkey": hotkey}
    if path:
        kwargs["path"] = path
    return bt.Wallet(**kwargs)


def unlock_coldkey(wallet: Any) -> bool:
    """Unlock coldkey with hardcoded password (non-interactive for PM2)."""
    pw = WALLET_PASSWORD
    try:
        if hasattr(wallet, "unlock_coldkey"):
            wallet.unlock_coldkey(password=pw)
            return True
        if hasattr(wallet, "coldkey_file"):
            wallet.coldkey_file.decrypt(pw)
            return True
        os.environ["BT_WALLET_PASSWORD"] = pw
        _ = wallet.coldkey
        return True
    except Exception as e:
        log(
            f"ERROR unlock coldkey {getattr(wallet, 'name', '?')}: "
            f"{type(e).__name__}: {e}"
        )
        return False


def is_registered(sub: Any, netuid: int, hotkey_ss58: str) -> bool:
    return bool(sub.is_hotkey_registered(netuid=netuid, hotkey_ss58=hotkey_ss58))


def get_uid(sub: Any, netuid: int, hotkey_ss58: str) -> Optional[int]:
    try:
        mg = sub.metagraph(netuid)
        return int(mg.hotkeys.index(hotkey_ss58))
    except Exception:
        return None


def coldkey_balance_tao(sub: Any, wallet: Any) -> Optional[float]:
    try:
        bal = sub.get_balance(wallet.coldkeypub.ss58_address)
        if hasattr(bal, "tao"):
            return float(bal.tao)
        return float(bal) / RAO_PER_TAO
    except Exception as e:
        log(f"WARN  balance check failed: {type(e).__name__}: {e}")
        return None


def reg_cost_tao(sub: Any, netuid: int) -> Optional[float]:
    for attr in ("recycle", "burn", "neuron_registration_cost"):
        fn = getattr(sub, attr, None)
        if callable(fn):
            try:
                val = fn(netuid=netuid)
                if hasattr(val, "tao"):
                    return float(val.tao)
                return float(val) / RAO_PER_TAO
            except Exception:
                continue
    return None


def pm2_restart(name: str, dry_run: bool) -> bool:
    cmd = ["pm2", "restart", name, "--update-env"]
    if dry_run:
        log(f"DRY   would run: {' '.join(cmd)}")
        return True
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            log(
                f"ERROR pm2 restart {name}: rc={r.returncode} "
                f"stderr={r.stderr.strip()[:400]}"
            )
            return False
        log(f"OK    pm2 restart {name}")
        return True
    except Exception as e:
        log(f"ERROR pm2 restart {name}: {type(e).__name__}: {e}")
        return False


def _wallet_label(wallet: Any, hotkey_name: str) -> str:
    name = getattr(wallet, "name", "?")
    return f"{name}/{hotkey_name}"


def burned_register(
    sub: Any, wallet: Any, netuid: int, hotkey_name: str, dry_run: bool
) -> bool:
    label = _wallet_label(wallet, hotkey_name)
    if dry_run:
        log(f"DRY   would burned_register wallet={label} netuid={netuid}")
        return True
    if not unlock_coldkey(wallet):
        return False
    try:
        ok = sub.burned_register(
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        if ok is False:
            log(f"ERROR burned_register returned False for {label}")
            return False
        log(f"OK    burned_register submitted for {label}")
        return True
    except Exception as e:
        log(f"ERROR burned_register {label}: {type(e).__name__}: {e}")
        return False


def wait_registered(
    sub: Any, netuid: int, hotkey_ss58: str, timeout_s: float = 180.0
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if is_registered(sub, netuid, hotkey_ss58):
                return True
        except Exception as e:
            log(f"WARN  wait_registered poll: {type(e).__name__}: {e}")
        time.sleep(5.0)
    return False


def try_reregister(
    *,
    bt: Any,
    sub: Any,
    miner: MinerSpec,
    netuid: int,
    wallet_path: Optional[str],
    dry_run: bool,
    cooldown_until: dict[str, float],
    cooldown_s: float,
) -> bool:
    """Attempt register+restart for one miner. Returns True if an attempt was made."""
    key = miner.pm2
    now = time.time()
    if now < cooldown_until.get(key, 0.0):
        left = int(cooldown_until[key] - now)
        log(f"SKIP  {miner.pm2} in cooldown ({left}s left)")
        return False

    wallet = make_wallet(bt, miner.wallet, miner.hotkey, wallet_path)
    hotkey_ss58 = wallet.hotkey.ss58_address

    log(
        f"ACTION re-registering {miner.pm2} "
        f"wallet={miner.wallet} hotkey={miner.hotkey} ss58={hotkey_ss58}"
    )

    cost = reg_cost_tao(sub, netuid)
    bal = coldkey_balance_tao(sub, wallet)
    if cost is not None and bal is not None and bal < cost:
        log(
            f"ERROR {miner.pm2} insufficient balance: "
            f"balance={bal:.6f} τ < cost≈{cost:.6f} τ — will retry later"
        )
        cooldown_until[key] = now + cooldown_s
        return True  # counted as this cycle's attempt slot
    if cost is not None:
        log(f"INFO  {miner.pm2} reg_cost≈{cost:.6f} τ balance={bal}")

    if not burned_register(
        sub, wallet, netuid, miner.hotkey, dry_run=dry_run
    ):
        cooldown_until[key] = now + cooldown_s
        return True

    if dry_run:
        pm2_restart(miner.pm2, dry_run=True)
        return True

    if not wait_registered(sub, netuid, hotkey_ss58):
        log(
            f"ERROR {miner.pm2} register submitted but still not on metagraph after wait"
        )
        cooldown_until[key] = now + cooldown_s
        return True

    uid = get_uid(sub, netuid, hotkey_ss58)
    log(f"OK    {miner.pm2} re-registered uid={uid}")
    pm2_restart(miner.pm2, dry_run=False)
    return True


def run_cycle(
    *,
    bt: Any,
    sub: Any,
    miners: list[MinerSpec],
    netuid: int,
    wallet_path: Optional[str],
    dry_run: bool,
    cooldown_until: dict[str, float],
    cooldown_s: float,
) -> None:
    """Check all miners; re-register at most ONE deregistered miner this cycle."""
    log(f"--- cycle start: checking {len(miners)} miners on netuid={netuid} ---")

    deregistered: list[MinerSpec] = []
    ok_count = 0
    err_count = 0

    for miner in miners:
        wallet = make_wallet(bt, miner.wallet, miner.hotkey, wallet_path)
        try:
            hotkey_ss58 = wallet.hotkey.ss58_address
            registered = is_registered(sub, netuid, hotkey_ss58)
        except Exception as e:
            err_count += 1
            log(f"ERROR {miner.pm2} check: {type(e).__name__}: {e}")
            continue

        if registered:
            ok_count += 1
            uid = get_uid(sub, netuid, hotkey_ss58)
            log(
                f"OK    {miner.pm2} registered "
                f"wallet={miner.wallet} hotkey={miner.hotkey} uid={uid}"
            )
        else:
            deregistered.append(miner)
            log(
                f"ALERT {miner.pm2} DEREGISTERED "
                f"wallet={miner.wallet} hotkey={miner.hotkey} ss58={hotkey_ss58}"
            )

    log(
        f"INFO  summary: registered={ok_count} deregistered={len(deregistered)} "
        f"errors={err_count}"
    )

    if not deregistered:
        log("--- cycle done (nothing to register) ---")
        return

    # Only one register per 10-minute cycle.
    target = deregistered[0]
    waiting = [m.pm2 for m in deregistered[1:]]
    if waiting:
        log(
            f"INFO  registering only {target.pm2} this cycle; "
            f"{len(waiting)} more waiting: {', '.join(waiting)}"
        )
    else:
        log(f"INFO  registering {target.pm2} this cycle")

    try_reregister(
        bt=bt,
        sub=sub,
        miner=target,
        netuid=netuid,
        wallet_path=wallet_path,
        dry_run=dry_run,
        cooldown_until=cooldown_until,
        cooldown_s=cooldown_s,
    )
    log("--- cycle done ---")


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "SN45 auto-register watchdog "
            "(check all → burned_register at most 1 per 10m → pm2 restart)"
        )
    )
    p.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    p.add_argument("--network", default=DEFAULT_NETWORK, help="finney / local / ws endpoint")
    p.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="Seconds between checks (default 600 = 10 min)",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help="Seconds to wait after a failed register before retrying that miner",
    )
    p.add_argument("--wallet.path", dest="wallet_path", default=None)
    p.add_argument(
        "--config",
        default=None,
        help='Optional JSON with {"miners":[{"pm2","wallet","hotkey"},...]}',
    )
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Check only; do not register or pm2 restart",
    )
    args = p.parse_args()

    import bittensor as bt

    miners = load_miners(args.config)
    log(
        f"Starting SN45 auto-register watchdog  netuid={args.netuid}  "
        f"network={args.network}  miners={len(miners)}  "
        f"interval={args.interval}s  max_register_per_cycle=1  "
        f"dry_run={args.dry_run}"
    )

    cooldown_until: dict[str, float] = {}

    while True:
        try:
            try:
                sub = bt.Subtensor(network=args.network)
            except Exception as e:
                log(f"WARN  subtensor connect: {type(e).__name__}: {e}")
                if args.once:
                    return 1
                time.sleep(max(5.0, args.interval))
                continue
            run_cycle(
                bt=bt,
                sub=sub,
                miners=miners,
                netuid=args.netuid,
                wallet_path=args.wallet_path,
                dry_run=args.dry_run,
                cooldown_until=cooldown_until,
                cooldown_s=args.cooldown,
            )
        except Exception as e:
            log(f"ERROR cycle: {type(e).__name__}: {e}")
        if args.once:
            return 0
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("stopped")
        raise SystemExit(0)
