'''
Class for binaural sounds (i.e. with 2 channels).
This module uses doctests. Use like so:
python -m doctest binaural.py
'''

import copy
import numpy

import slab.sound
import slab.signals
Sound = slab.sound.Sound
Signal = slab.signals

class Binaural(Sound):
	# TODO: Add reverb (image model) room simulation.
	'''
	Class for working with binaural sounds, including ITD and ILD manipulation. Binaural inherits all signal generation functions from the Sound class, but returns binaural signals. Recasting an object of class sound or signal with 1 or 3+ channels calls Sound.copychannel to return a binaural sound with two channels identical to the first channel of the original signal.
	Properties:
	Binaural.left: left (0th)channel
	Binaural.right: right (1st) channel

	>>> sig = Binaural.whitenoise()
	>>> sig.nchannels
	2
	>>> all(sig.left - sig.right)
	False
	'''

	# instance properties
	def _set_left(self, other):
		if isinstance(other, Sound):
			self.data[:, 0] = other.data[:, 0]
		else:
			self.data[:, 0] = numpy.array(other)

	def _set_right(self, other):
		if isinstance(other, Sound):
			self.data[:, 1] = other.data[:, 0]
		else:
			self.data[:, 1] = numpy.array(other)

	left = property(fget=lambda self: self.channel(0), fset=_set_left,
					doc='The left channel for a stereo sound.')
	right = property(fget=lambda self: self.channel(1), fset=_set_right,
					 doc='The right channel for a stereo sound.')


	def __init__(self, data, samplerate=None):
		if isinstance(data, (slab.sound.Sound, slab.signals.Signal)):
			if data.nchannels != 2: # TODO: doesn't result in 2 channels!!!
				data.copychannel(2)
			self.data = data.data
			self.samplerate = data.samplerate
		elif isinstance(data, (list, tuple)):
			if isinstance(data[0], (slab.sound.Sound, slab.signals.Signal)):
				if data[0].nsamples != data[1].nsamples:
					ValueError('Sounds must have same number of samples!')
				if data[0].samplerate != data[1].samplerate: # TODO: This doesn't catch for some reason!
					ValueError('Sounds must have same samplerate!')
				super().__init__((data[0].data[:,0], data[1].data[:,0]), data[0].samplerate)
			else:
				super().__init__(data, samplerate)
		else:
			super().__init__(data, samplerate)
			if self.nchannels != 2:
				ValueError('Binaural sounds must have two channels!')

	def itd(self, duration=600e-6):
		'''
		Returns a binaural sound object with one channel delayed with respect to the other channel by duration (*600 microseconds*), which can be the number of samples or a length of time in seconds.
		Negative dB values delay the right channel (virtual sound source moves to the left). itd requires a sound with two channels.
		>>> sig = Binaural.whitenoise()
		>>> _ = sig.itd(1)
		>>> _ = sig.itd(-0.001)

		'''
		duration = Sound.in_samples(duration, self.samplerate)
		new = copy.deepcopy(self) # so that we can return a new signal
		if duration == 0: return new # nothing needs to be shifted
		if duration < 0: # negative itds by convention shift to the left (i.e. delay right channel)
			channel = 1 # right
		else:
			channel = 0 # left
		new.delay(duration=abs(duration), chan=channel)
		return new

	def ild(self, dB):
		'''
		Returns a sound object with one channel attenuated with respect to
		the other channel by dB. Negative dB values attenuate the right channel
		(virtual sound source moves to the left). The mean intensity of the signal
		is kept constant.
		ild requires a sound with two channels.
		>>> sig = Binaural.whitenoise()
		>>> _ = sig.ild(3)
		>>> _ = sig.ild(-3)

		'''
		new = copy.deepcopy(self) # so that we can return a new signal
		level = numpy.mean(self.level)
		new_levels = (level - dB/2, level + dB/2)
		new.level = new_levels
		return new

	def itd_ramp(self, from_itd, to_itd):
		'''
		Returns a sound object with a linearly increasing or decreasing interaural time difference. This is achieved by sinc interpolation of one channel
		with a dynamic delay. The resulting virtual sound source moves to the left or right. from_itd and to_itd are the itd values at the beginning and end
		of the sound. Delays in between are linearely interpolated. moving_ild requires a sound with two channels.
		>>> sig = Binaural.whitenoise()
		>>> _ = sig.itd_ramp(from_itd=-0.001, to_itd=0.01)

		'''
		new = copy.deepcopy(self)
		# make the ITD ramps
		left_ramp = numpy.linspace(-from_itd/2, -to_itd/2, self.nsamples)
		right_ramp = numpy.linspace(from_itd/2, to_itd/2, self.nsamples)
		if self.nsamples >= 8192:
			filter_length = 1024
		elif self.nsamples >= 512:
			filter_length = self.nsamples//16*2 # 1/8th of nsamples, always even
		else:
			ValueError('Signal too short! (min 512 samples)')
		new.delay(duration=left_ramp, chan=0, filter_length=filter_length)
		new.delay(duration=right_ramp, chan=1, filter_length=filter_length)
		return new

	def ild_ramp(self, from_ild, to_ild):
		'''
		Returns a sound object with a linearly increasing or decreasing interaural level difference. The resulting virtual sound source moves to the left
		or right. from_ild and to_ild are the itd values at the beginning and end of the sound. ILDs in between are linearely interpolated. moving_ild requires a sound with two channels.
		>>> sig = Binaural.whitenoise()
		>>> move = sig.ild_ramp(from_ild=-50, to_ild=50)
		>>> move.play()
		'''
		new = self.ild(0) # set ild to zero
		# make ramps
		left_ramp = numpy.linspace(-from_ild/2, -to_ild/2, self.nsamples)
		right_ramp = numpy.linspace(from_ild/2, to_ild/2, self.nsamples)
		left_ramp = 10**(left_ramp/20.)
		right_ramp = 10**(right_ramp/20.)
		# multiply channels with ramps
		new.data[:,0] *= left_ramp
		new.data[:,1] *= right_ramp
		return new

	@staticmethod
	def whitenoise(duration=1.0, kind='diotic', samplerate=None, normalise=True):
		'''
		Returns a white noise. If the samplerate is not specified, the global default value will be used. kind = 'diotic' produces the same noise samples in both channels, kind = 'dichotic' produces uncorrelated noise.
		>>> noise = Binaural.whitenoise(kind='diotic')
		'''
		if kind == 'dichotic':
			out = Binaural(Sound.whitenoise(duration=duration, nchannels=2, samplerate=samplerate, normalise=normalise))
		elif kind == 'diotic':
			out = Binaural(Sound.whitenoise(duration=duration, nchannels=2, samplerate=samplerate, normalise=normalise))
			out.left = out.right
		return out

	@staticmethod
	def pinknoise(duration=1.0, kind='diotic', samplerate=None, normalise=True):
		'''
		Returns a pink noise. If the samplerate is not specified, the global default value will be used. kind = 'diotic' produces the same noise samples in both channels, kind = 'dichotic' produces uncorrelated noise.
		>>> noise = Binaural.pinknoise(kind='diotic')
		'''
		if kind == 'dichotic':
			out = Binaural(Sound.powerlawnoise(duration, 1.0, samplerate=samplerate, nchannels=2, normalise=normalise))
		elif kind == 'diotic':
			out = Binaural(Sound.powerlawnoise(duration, 1.0, samplerate=samplerate, nchannels=2, normalise=normalise))
			out.left = out.right
		return out

if __name__ == '__main__':
	sig = slab.Binaural.pinknoise(duration=0.5, samplerate=44100)
	sig.filter(kind='bp',f=[100,6000])
	sig.ramp(when='both',duration=0.15)
	sig_itd = sig.itd_ramp(500e-6,-500e-6)
	sig_itd.play()
	
