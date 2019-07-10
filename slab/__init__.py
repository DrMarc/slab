# install from github:
# pip install git+git://github.com/DrMarc/soundtools.git

import pathlib
import os
DATAPATH = str(pathlib.Path(__file__).parent.parent.resolve() / pathlib.Path('data')) + os.sep

from slab.signals import *
from slab.sound import *
from slab.binaural import *
from slab.psychoacoustics import *
from slab.hrtf import *
