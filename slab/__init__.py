import pathlib
import os
import sys

__version__ = '0.8.0'

sys.path.append('..\\')
DATAPATH = str(pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')) + os.sep

from slab.hrtf import *
from slab.psychoacoustics import *
from slab.binaural import *
from slab.sound import *
from slab.signal import *
