"""Run bounded numerical verification from the source folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from face2ceph.reproduction import main


if __name__ == "__main__":
    raise SystemExit(main())
