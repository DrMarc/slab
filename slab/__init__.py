import pathlib
import os
import sys

__version__ = '0.8.0'

sys.path.append('..\\')
DATAPATH = str(pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')) + os.sep

from slab.hrtf import HRTF
from slab.psychoacoustics import *
from slab.binaural import Binaural
from slab.sound import Sound, set_default_level
from slab.signal import Signal, set_default_samplerate
