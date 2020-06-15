# install from github:
# pip install git+git://github.com/DrMarc/soundtools.git

from slab.hrtf import *
from slab.psychoacoustics import *
from slab.binaural import *
from slab.sound import *
from slab.signals import *
import pathlib
import os
DATAPATH = str(pathlib.Path(__file__).parent.resolve() / pathlib.Path('data')) + os.sep
