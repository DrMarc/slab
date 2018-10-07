'''
Class for HRTFs and microphone/speaker filter functions
'''

# methods pertaining to both should go here:
# TODO: calculating/plotting transfer function
# TODO: applying the filter to a Sound
# TODO: inverse filter
# TODO: making standard filters (high, low, notch, bp)
# TODO: FIR, but also FFR filters?
# TODO: add gammatone filterbank

from slab.signals import Signal # getting the base class

class Filter(Signal):
	pass
