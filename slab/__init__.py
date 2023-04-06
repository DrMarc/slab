import sys
import os
import pathlib

__version__ = '1.1.3'

spyder_env = set(['SPYDER_ARGS',
                  'SPY_EXTERNAL_INTERPRETER',
                  'SPY_UMR_ENABLED',
                  'SPY_UMR_VERBOSE',
                  'SPY_UMR_NAMELIST',
                  'SPY_RUN_LINES_O',
                  'SPY_PYLAB_O',
                  'SPY_BACKEND_O',
                  'SPY_AUTOLOAD_PYLAB_O',
                  'SPY_FORMAT_O',
                  'SPY_RESOLUTION_O',
                  'SPY_WIDTH_O',
                  'SPY_HEIGHT_O',
                  'SPY_USE_FILE_O',
                  'SPY_RUN_FILE_O',
                  'SPY_AUTOCALL_O',
                  'SPY_GREEDY_O',
                  'SPY_SYMPY_O',
                  'SPY_RUN_CYTHON',
                  'SPYDER_PARENT_DIR'])
if "JPY_PARENT_PID" in os.environ:
    overlap = spyder_env & set(os.environ.keys())
    if spyder_env.intersection(set(os.environ.keys())):
        in_notebook = True

sys.path.append('..\\')

from slab.hrtf import HRTF
from slab.psychoacoustics import *
from slab.binaural import Binaural
from slab.sound import Sound, set_default_level, set_calibration_intensity, get_calibration_intensity, calibrate
from slab.signal import Signal, set_default_samplerate, get_default_samplerate
from slab.filter import Filter

def cite(fmt='bibtex'):
    """
    Return the citation string of the slab module.

    Arguments:
        fmt (str): if bibtex, return the bibtex citation string, otherwise return the text reference
    """
    if fmt == 'bibtex':
        return """@article{Schönwiesner2021, doi = {10.21105/joss.03284}, url = {https://doi.org/10.21105/joss.03284}, year = {2021}, publisher = {The Open Journal}, volume = {6}, number = {62}, pages = {3284}, author = {Marc Schönwiesner and Ole Bialas}, title = {s(ound)lab: An easy to learn Python package for designing and running psychoacoustic experiments.}, journal = {Journal of Open Source Software}}"""
    return """Schönwiesner et al., (2021). s(ound)lab: An easy to learn Python package for designing and running psychoacoustic experiments. Journal of Open Source Software, 6(62): 3284, https://doi.org/10.21105/joss.03284"""
