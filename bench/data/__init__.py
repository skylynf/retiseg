"""Readers for datasets kept in their original release layout.

Evaluation still uses the directory contract in bench/common/io.py.
These readers address the files under dataset/ without copying them.
"""

from bench.data.catalog import catalog
from bench.data.ddr import open_ddr
from bench.data.diaretdb import open_diaretdb0, open_diaretdb1
from bench.data.idrid import open_idrid
from bench.data.tjdr import open_tjdr

__all__ = ["catalog", "open_idrid", "open_ddr", "open_diaretdb0", "open_diaretdb1", "open_tjdr"]
