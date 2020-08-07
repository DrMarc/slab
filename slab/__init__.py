
import pathlib
import os
from _version import __version__
DATAPATH = str(pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')) + os.sep
from slab.hrtf import *
from slab.psychoacoustics import *
from slab.binaural import *
from slab.sound import *
from slab.signals import *
