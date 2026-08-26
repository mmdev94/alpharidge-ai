#!/usr/bin/env python3
"""Watch SN45 miner hotkeys; auto-register via btcli + pm2 restart on deregistration.

Polls the chain every 1 hour. Checks all miners; if any are deregistered,
re-registers **at most one** per cycle with ``btcli subnet register``, then
``pm2 restart`` that miner.

Also restarts ``alpha-pool`` once per day at **00:00 UTC** (pool-only; miners
stay up so IsAlive keeps working).

Usage (from repo root)::

  pm2 start .venv/bin/python --name alpha-reg-watch -- \\
    watch_sn45_auto_register.py --network finney

  .venv/bin/python watch_sn45_auto_register.py --once --dry-run

Create ``~/.bittensor/password.txt`` (one line, coldkey password) before enabling
live register. Override with ``--password-file`` if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_NETUID = 45
DEFAULT_NETWORK = "finney"
DEFAULT_INTERVAL_S = 3600.0  # 1 hour
DEFAULT_COOLDOWN_S = 3600.0  # skip same miner for 1h after a failed register
RAO_PER_TAO = 1_000_000_000

# Coldkey password file for btcli --wallet-password-file (create this yourself).
DEFAULT_PASSWORD_FILE = str(Path.home() / ".bittensor" / "password.txt")

# Daily pool restart at 00:00 UTC (memory reclaim; miners stay running).
DEFAULT_POOL_PM2 = "alpha-pool"
POOL_RESTART_POLL_S = 30.0

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


def resolve_btcli(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BTCLI")
    if env:
        return env
    here = Path(__file__).resolve().parent
    venv_btcli = here / ".venv" / "bin" / "btcli"
    if venv_btcli.is_file():
        return str(venv_btcli)
    found = shutil.which("btcli")
    if found:
        return found
    return "btcli"


def make_wallet(bt: Any, name: str, hotkey: str, path: Optional[str]) -> Any:
    kwargs: dict[str, Any] = {"name": name, "hotkey": hotkey}
    if path:
        kwargs["path"] = path
    return bt.Wallet(**kwargs)


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


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def maybe_restart_pool_daily(
    *,
    pool_pm2: Optional[str],
    last_restart_date: date,
    dry_run: bool,
) -> date:
    """Restart pool once when UTC date rolls over (≈00:00 UTC). Returns updated date."""
    if not pool_pm2:
        return last_restart_date
    today = utc_today()
    if today <= last_restart_date:
        return last_restart_date
    log(
        f"ACTION daily {pool_pm2} restart (UTC date {last_restart_date} → {today})"
    )
    pm2_restart(pool_pm2, dry_run=dry_run)
    return today


def sleep_between_cycles(
    interval_s: float,
    *,
    pool_pm2: Optional[str],
    last_pool_restart_date: date,
    dry_run: bool,
) -> date:
    """Sleep ``interval_s``, polling near 00:00 UTC for daily pool restart."""
    deadline = time.time() + max(5.0, interval_s)
    last = last_pool_restart_date
    while True:
        last = maybe_restart_pool_daily(
            pool_pm2=pool_pm2,
            last_restart_date=last,
            dry_run=dry_run,
        )
        left = deadline - time.time()
        if left <= 0:
            break
        time.sleep(min(POOL_RESTART_POLL_S, left))
    return last


def btcli_register(
    *,
    btcli: str,
    miner: MinerSpec,
    hotkey_ss58: str,
    netuid: int,
    network: str,
    wallet_path: Optional[str],
    password_file: str,
    dry_run: bool,
) -> bool:
    """Register via ``btcli subnet register`` (btcli 11.x needs explicit --hotkey)."""
    # btcli 11: --hotkey must be SS58 (or WALLET/HOTKEY); wallet-hotkey alone is not enough.
    cmd = [
        btcli,
        "subnet",
        "register",
        "--netuid",
        str(netuid),
        "--wallet",
        miner.wallet,
        "--wallet-hotkey",
        miner.hotkey,
        "--hotkey",
        hotkey_ss58,
        "--network",
        network,
        "--yes",
        "--wallet-password-file",
        password_file,
    ]
    if wallet_path:
        cmd.extend(["--wallet-path", wallet_path])

    if dry_run:
        log(f"DRY   would run: {' '.join(cmd)}")
        return True

    if not Path(password_file).is_file():
        log(
            f"ERROR password file missing: {password_file} "
            f"(create it with your coldkey password on one line)"
        )
        return False

    try:
        log(f"INFO  running: {' '.join(cmd)}")
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
        if out:
            for line in out.splitlines()[-40:]:
                log(f"btcli {line}")
        if r.returncode != 0:
            log(
                f"ERROR btcli register {miner.wallet}/{miner.hotkey} "
                f"rc={r.returncode}"
            )
            return False
        log(f"OK    btcli register {miner.wallet}/{miner.hotkey}")
        return True
    except Exception as e:
        log(
            f"ERROR btcli register {miner.wallet}/{miner.hotkey}: "
            f"{type(e).__name__}: {e}"
        )
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
    network: str,
    wallet_path: Optional[str],
    password_file: str,
    btcli: str,
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
        return True
    if cost is not None:
        log(f"INFO  {miner.pm2} reg_cost≈{cost:.6f} τ balance={bal}")

    if not btcli_register(
        btcli=btcli,
        miner=miner,
        hotkey_ss58=hotkey_ss58,
        netuid=netuid,
        network=network,
        wallet_path=wallet_path,
        password_file=password_file,
        dry_run=dry_run,
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
    network: str,
    wallet_path: Optional[str],
    password_file: str,
    btcli: str,
    dry_run: bool,
    cooldown_until: dict[str, float],
    cooldown_s: float,
) -> None:
    """Check all miners; re-register at most ONE deregistered miner this cycle."""
    log(
        f"--- cycle start: checking {len(miners)} miners on netuid={netuid} "
        f"(max 1 register / cycle) ---"
    )

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

    # Only one register per cycle (default every 1 hour).
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
        network=network,
        wallet_path=wallet_path,
        password_file=password_file,
        btcli=btcli,
        dry_run=dry_run,
        cooldown_until=cooldown_until,
        cooldown_s=cooldown_s,
    )
    log("--- cycle done ---")


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "SN45 auto-register watchdog "
            "(hourly check → btcli register at most 1 → pm2 restart miner; "
            "daily 00:00 UTC alpha-pool restart)"
        )
    )
    p.add_argument("--netuid", type=int, default=DEFAULT_NETUID)
    p.add_argument("--network", default=DEFAULT_NETWORK, help="finney / local / ws endpoint")
    p.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="Seconds between checks (default 3600 = 1 hour)",
    )
    p.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help="Seconds to wait after a failed register before retrying that miner",
    )
    p.add_argument("--wallet.path", dest="wallet_path", default=None)
    p.add_argument(
        "--password-file",
        default=DEFAULT_PASSWORD_FILE,
        help=f"Coldkey password file for btcli (default: {DEFAULT_PASSWORD_FILE})",
    )
    p.add_argument(
        "--btcli",
        default=None,
        help="Path to btcli (default: .venv/bin/btcli or PATH)",
    )
    p.add_argument(
        "--pool-pm2",
        default=DEFAULT_POOL_PM2,
        help=f"PM2 process to restart daily at 00:00 UTC (default: {DEFAULT_POOL_PM2})",
    )
    p.add_argument(
        "--no-pool-restart",
        action="store_true",
        help="Disable daily alpha-pool restart",
    )
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
    btcli = resolve_btcli(args.btcli)
    password_file = os.path.expanduser(args.password_file)
    pool_pm2 = None if args.no_pool_restart else args.pool_pm2
    # Skip restart on boot; only fire after the next UTC midnight.
    last_pool_restart_date = utc_today()
    log(
        f"Starting SN45 auto-register watchdog  netuid={args.netuid}  "
        f"network={args.network}  miners={len(miners)}  "
        f"interval={args.interval}s  max_register_per_cycle=1  "
        f"btcli={btcli}  password_file={password_file}  "
        f"daily_pool_restart={pool_pm2 or 'off'}@00:00UTC  "
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
                last_pool_restart_date = sleep_between_cycles(
                    args.interval,
                    pool_pm2=pool_pm2,
                    last_pool_restart_date=last_pool_restart_date,
                    dry_run=args.dry_run,
                )
                continue
            run_cycle(
                bt=bt,
                sub=sub,
                miners=miners,
                netuid=args.netuid,
                network=args.network,
                wallet_path=args.wallet_path,
                password_file=password_file,
                btcli=btcli,
                dry_run=args.dry_run,
                cooldown_until=cooldown_until,
                cooldown_s=args.cooldown,
            )
        except Exception as e:
            log(f"ERROR cycle: {type(e).__name__}: {e}")
        if args.once:
            return 0
        last_pool_restart_date = sleep_between_cycles(
            args.interval,
            pool_pm2=pool_pm2,
            last_pool_restart_date=last_pool_restart_date,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("stopped")
        raise SystemExit(0)
