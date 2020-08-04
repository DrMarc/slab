import sys
from pathlib import Path
_location = Path(__file__).resolve().parents[2]  # append the main folder
sys.path.append(_location)


def test_generating_sounds():
    import slab
    sig = slab.Sound.tone()
