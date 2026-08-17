"""PM2 entry: shared inference pool on local port 30000 (default).

  pm2 start .venv/bin/python --name alpha-pool -- -m neurons.inference_pool

Env (optional):
  INFERENCE_POOL_HOST=127.0.0.1
  INFERENCE_POOL_PORT=30000
  INFERENCE_POOL_WORKERS=10
"""

from __future__ import annotations

import os

# Load .miner_env before pool imports config-heavy analyzers
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from alpharidge_ai.inference_pool import run_server


def main():
    host = os.getenv("INFERENCE_POOL_HOST", "127.0.0.1")
    port = int(os.getenv("INFERENCE_POOL_PORT", "30000"))
    workers = int(os.getenv("INFERENCE_POOL_WORKERS", "10"))
    run_server(host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
