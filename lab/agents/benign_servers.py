"""Long-running listeners for the benign background traffic (lab/agents/noise.py's clients need
something to talk to). Named separately from `noise` on purpose: generate_lab_runs.py kills the
noise *clients* between sessions with `pkill -f lab.agents.noise`, which must not kill these."""

from __future__ import annotations

import time

from lab.agents.noise import start_servers


def main() -> int:
    start_servers()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    raise SystemExit(main())
