import slab
import scipy
from pprint import pprint

monaural = slab.Sound.tone()

binaural = slab.Binaural(monaural)

binaural.drr()