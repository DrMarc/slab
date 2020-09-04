
from slab.signal import *
from slab.sound import *
from slab.binaural import *
from slab.psychoacoustics import *
from slab.hrtf import *
from _version import __version__
import pathlib
import os
import sys
sys.path.append('..\\')
DATAPATH = str(pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')) + os.sep
