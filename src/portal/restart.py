"""Ask the running portal to restart as soon as no job is running.

Usage::

    uv run python -m portal.restart           # request a restart
    uv run python -m portal.restart --cancel  # withdraw the request

The portal checks every few seconds. When no job is uploading, queued or
running, it exits with a special code and ``start-portal.ps1`` starts it again
with the new code. The tunnel keeps running, so the public link stays the
same. Jobs waiting for parent labels are kept and can continue afterwards.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from portal import app as portal_app


def main(argv: Sequence[str] | None = None) -> None:
    """Write or remove the restart request.

    Args:
        argv: Command-line arguments (``--cancel`` to withdraw the request).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cancel", action="store_true", help="Withdraw the request.")
    args = parser.parse_args(argv)
    if args.cancel:
        portal_app.restart_flag_path().unlink(missing_ok=True)
        print("Restart request withdrawn.")
        return
    portal_app.request_restart()
    if portal_app.portal_is_idle():
        print(
            "Restart requested. No job is running, so the portal restarts within ~10 s."
        )
    else:
        print(
            "Restart requested. A job is running; the portal will restart as soon "
            "as it finishes (jobs waiting for parent labels do not block it)."
        )


if __name__ == "__main__":
    main()
