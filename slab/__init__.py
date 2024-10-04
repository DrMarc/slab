import sys
import os
import pathlib

__version__ = '1.7.0'

# The variable _in_notebook is used to enable audio playing in Jupiter notebooks
# and on Google colab (see slab.sound.play())
try:
    shell = get_ipython().__class__.__module__
    if 'terminal' in shell:
        _in_notebook = False
    else:
        _in_notebook = True  # probably in a notebook
except NameError:
    _in_notebook = False  # probably standard Python interpreter

sys.path.append('..\\')

from slab.hrtf import HRTF, Room
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
