import copy
import warnings
import numpy
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib, plt = False, False
try:
    import scipy.signal
except ImportError:
    scipy = False

_default_samplerate = 8000  # default samplerate in Hz; used by all methods if on samplerate argument is provided.

def set_default_samplerate(samplerate):
    global _default_samplerate
    _default_samplerate = samplerate

def get_default_samplerate():
    global _default_samplerate
    return _default_samplerate

class Signal:
    """
    Base class for Signal data (from which the Sound and Filter class inherit).
    Provides arithmetic operations, slicing, and conversion between samples and times.

    Arguments:
        data (numpy.ndarray | slab.Signal | list): samples of the sound. If it is an array, the first dimension
            should represent the number of samples and the second one the number of channels. If it's an object,
            it must have a .data attribute containing an array. If it's a list, the elements can be arrays or objects.
            The output will be a multi-channel sound with each channel corresponding to an element of the list.
        samplerate (int | None): the samplerate of the sound. If None, use the default samplerate.
    Attributes:
        .duration: duration of the sound in seconds
        .n_samples: duration of the sound in samples
        .n_channels: number of channels in the sound
        .times: list with the time point of each sample
    Examples::

        import slab, numpy
        sig = slab.Signal(numpy.ones([10,2]),samplerate=10)  # create a sound
        sig[:5] = 0  # set the first 5 samples to 0
        sig[:,1]  # select the data from the second channel
        sig2 = sig * 2  # multiply each sample by 2
        sig_inv = -sig  # invert the phase
    """
    # instance properties
    n_samples = property(fget=lambda self: self.data.shape[0],
                         doc='The number of samples in the Signal.')
    duration = property(fget=lambda self: self.data.shape[0] / self.samplerate,
                        doc='The length of the Signal in seconds.')
    times = property(fget=lambda self: numpy.arange(self.data.shape[0], dtype=float) / self.samplerate,
                     doc='An array of times (in seconds) corresponding to each sample.')
    n_channels = property(fget=lambda self: self.data.shape[1],
                          doc='The number of channels in the Signal.')

    # __methods (class creation, printing, and slice functionality)
    def __init__(self, data, samplerate=None):
        if hasattr(data, 'samplerate') and samplerate is not None:
            warnings.warn('First argument has a samplerate property. Ignoring given samplerate.')
        if isinstance(data, numpy.ndarray):
            self.data = numpy.array(data, dtype='float')
        elif isinstance(data, (list, tuple)):
            if all([hasattr(c, 'data') and hasattr(c, 'samplerate') for c in data]):  # all slab objects
                if not all(c.samplerate == data[0].samplerate for c in data):
                    raise ValueError('All elements of list must have the same samplerate!')
                if not all(c.n_samples == data[0].n_samples for c in data):
                    raise ValueError('All elements of list must have the same number of samples!')
                channel_data = [c.data for c in data]  # all clear, now get channel arrays
                self.data = numpy.hstack(channel_data)
                self.samplerate = data[0].samplerate
            elif all([isinstance(c, numpy.ndarray) for c in data]):
                channel_data = tuple(Signal(c, samplerate=samplerate) for c in data)
                self.data = numpy.hstack(channel_data)
        # any object with data and samplerate attributes can be recast as Signal
        elif hasattr(data, 'data') and hasattr(data, 'samplerate'):
            self.data = data.data
            self.samplerate = data.samplerate
        else:
            raise TypeError('Cannot initialise Signal with data of class ' + str(data.__class__))
        if self.data.ndim == 1:
            self.data = self.data[:,numpy.newaxis]
        elif self.data.shape[1] > self.data.shape[0]:
            if not len(data) == 0:  # dont transpose if data is an empty array
                self.data = self.data.T
        if not hasattr(self, 'samplerate'):  # if samplerate has not been set, use default
            if samplerate is None:
                self.samplerate = _default_samplerate
            else:
                self.samplerate = samplerate

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)}\n{repr(self.samplerate)})'

    def __str__(self):
        return f'{type(self)} duration {self.duration}, samples {self.n_samples}, channels {self.n_channels},' \
               f'samplerate {self.samplerate}'

    def _repr_html_(self):
        'HTML image representation for Jupyter notebook support'
        elipses = '\u2026'
        class_name = str(type(self))[8:-2]
        html = [f'<h4>{class_name} with samplerate = {self.samplerate}</h4>']
        html += ['<table><tr><th>#</th>']
        samps, chans = self.data.shape
        html += (f'<th>channel {j}</th>' for j in range(chans))
        if samps > 7:
            rows = [0, 1, 2, -1, samps-3, samps-2, samps-1]  # -1 will output '...'
        else:
            rows = range(samps)
        for i in rows:
            html.append(f'<tr><th>{elipses if i == -1 else i}</th>')
            for j in range(chans):
                if i == -1:
                    html.append(f'<td>{elipses}</td>')
                else:
                    html.append(f'<td>{self.data[i, j]:.5f}</td>')
        html.append('</table>')
        return ''.join(html)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    # arithmetic operators
    def __add__(self, other):
        new = copy.deepcopy(self)
        if isinstance(other, type(self)):
            new.data = self.data + other.data
        else:
            new.data = self.data + other
        return new
    __radd__ = __add__

    def __sub__(self, other):
        new = copy.deepcopy(self)
        if isinstance(other, type(self)):
            new.data = self.data - other.data
        else:
            new.data = self.data - other
        return new
    __rsub__ = __sub__

    def __mul__(self, other):
        new = copy.deepcopy(self)
        if isinstance(other, type(self)):
            new.data = self.data * other.data
        else:
            new.data = self.data * other
        return new
    __rmul__ = __mul__

    def __truediv__(self, other):
        new = copy.deepcopy(self)
        if isinstance(other, type(self)):
            new.data = self.data / other.data
        else:
            new.data = self.data / other
        return new
    __rtruediv__ = __truediv__

    def __pow__(self, other):
        new = copy.deepcopy(self)
        new.data = self.data ** other
        return new

    def __neg__(self):
        new = copy.deepcopy(self)
        new.data = self.data*-1
        return new

    def __len__(self):
        return self.n_samples

    # static methods (belong to the class, but can be called without creating an instance)
    @staticmethod
    def in_samples(ctime, samplerate):
        """
        Converts time values in seconds to samples. This is used to enable input in either samples (integers) or
        seconds (floating point numbers) in the class.

        Arguments:
            ctime (int | float | list | numpy.ndarray): the time(s) to convert to samples.
            samplerate (int): the samplerate of the sound.
        Returns:
             (int | list | numpy.ndarray): the time(s) in samples.
        """
        out = None
        if isinstance(ctime, (int, numpy.int64)):  # single int is treated as samples
            out = ctime
        elif isinstance(ctime, (float, numpy.float64)):
            out = int(numpy.rint(ctime * samplerate))
        elif isinstance(ctime, numpy.ndarray):
            if isinstance(ctime[0], numpy.int64):
                out = ctime
            elif isinstance(ctime[0], numpy.float64):
                out = numpy.int64(numpy.rint(ctime * samplerate))
            else:
                ValueError('Unsupported type in numpy.ndarray (must be int64 of float 64)')
        elif isinstance(ctime, (list, tuple)):  # convert one by one
            out = numpy.empty_like(ctime)
            for i, c in enumerate(ctime):
                out[i] = numpy.int64(Signal.in_samples(c, samplerate))
        else:
            ValueError(
                'Unsupported type for ctime (must be int, float, numpy.ndarray of ints or floats, list or tuple)!')
        return out

    @staticmethod
    def set_default_samplerate(samplerate):
        """ Sets the global default samplerate for Signal objects, by default 8000 Hz. """
        global _default_samplerate
        _default_samplerate = samplerate
        return _default_samplerate

    # instance methods (belong to instances created from the class)
    def channel(self, n):
        """
        Get a single data channel.

        Arguments:
            n (int): channel index
        Returns:
            (slab.Signal): a new instance of the class that contains the selected channel as data.
        """
        new = copy.deepcopy(self)
        new.data = self.data[:, n]
        new.data.shape = (len(new.data), 1)
        return new

    def channels(self):
        """ Returns generator that yields channel data as objects of the calling class. """
        return (self.channel(i) for i in range(self.n_channels))

    def resize(self, duration):
        """
        Change the duration by padding with zeros or cutting the data.

        Arguments:
            duration (float | int): new duration of the sound in seconds (given a float) or in samples (given an int).
        Returns:
            (slab.Signal): a new instance of the same class with the specified duration.
        """
        duration = Signal.in_samples(duration, self.samplerate)
        resized = copy.deepcopy(self)
        if duration == len(self.data):
            pass  # already correct size
        elif duration < len(self.data):  # cut the data
            resized.data = self.data[:duration, :]
        else:  # pad with zeros
            padding = numpy.zeros((duration - len(self.data), self.n_channels))
            resized.data = numpy.concatenate((self.data, padding))
        return resized

    def trim(self, start=0, stop=None):
        """ Trim the signal by returning the section between `start` and `stop`.
        Arguments:
            start (float | int): start of the section in seconds (given a float) or in samples (given an int).
            stop (float | int): end of the section in seconds (given a float) or in samples (given an int).
        Returns:
            (slab.Signal): a new instance of the same class with the specified duration. """
        start = Signal.in_samples(start, self.samplerate)
        if stop is None:
            stop = self.n_samples - 1
        else:
            stop = Signal.in_samples(stop, self.samplerate)
        if stop >= self.n_samples:
            stop = self.n_samples - 1
        if start < 0:
            start = 0
        if start >= stop:
            raise ValueError('Start must precede stop.')
        trimmed = copy.deepcopy(self)
        trimmed.data = self.data[start:stop, :]
        return trimmed

    def resample(self, samplerate):
        """
        Resample the sound.

        Arguments:
            samplerate (int): the samplerate of the resampled sound.
        Returns:
            (slab.Signal): a new instance of the same class with the specified samplerate.
        """
        if scipy is False:
            raise ImportError('Resampling requires scipy.sound.')
        if self.samplerate == samplerate:
            return self
        out = copy.deepcopy(self)
        new_n_samples = int(numpy.rint(samplerate*self.duration))
        new_signal = numpy.zeros((new_n_samples, self.n_channels))
        for chan in range(self.n_channels):
            new_signal[:, chan] = scipy.signal.resample(
                self.channel(chan), new_n_samples).flatten()
        out.data = new_signal
        out.samplerate = samplerate
        return out

    def envelope(self, apply_envelope=None, times=None, kind='gain', cutoff=50):
        """
        Either apply an envelope to a sound or, if no `apply_envelope` was specified, compute the Hilbert envelope
        of the sound.

        Arguments:
            apply_envelope (None | numpy.ndarray): data to multiply with the sound. The envelope is linearly
                interpolated to be the same length as the sound. If None, compute the sound's Hilbert envelope
            times (None | numpy.ndarray | list): If None a vector linearly spaced from 0 to the duration of the sound
                is used. If time points (in seconds, clamped to the the sound duration) for the amplitude values
                in envelope are supplied, then the interpolation is piecewise linear between pairs of time and envelope
                valued (must have same length).
            kind (str): determines the unit of the envelope value
            cutoff (int): Frequency of the lowpass filter that is applied to remove the temporal fine structure in Hz.
        Returns:
            (slab.Signal): Either a copy of the instance with the specified envelope applied or the sound's
                Hilbert envelope.
        """
        if apply_envelope is None:  # get the signals envelope
            return self._get_envelope(kind, cutoff)
        else:  # apply the envelope to the sound
            return self._apply_envelope(apply_envelope, times, kind)

    def _get_envelope(self, kind, cutoff):
        if scipy is False:
            raise ImportError('Calculating envelopes requires scipy.signal.')
        envs = numpy.abs(scipy.signal.hilbert(self.data, axis=0))
        # 50Hz lowpass filter to remove fine-structure
        filt = scipy.signal.firwin(1000, cutoff, pass_zero=True, fs=self.samplerate)
        envs = scipy.signal.filtfilt(filt, [1], envs, axis=0)
        envs[envs <= 0] = numpy.finfo(float).eps  # remove negative values and zeroes
        if kind == 'dB':
            envs = 20 * numpy.log10(envs)  # convert amplitude to dB
        elif not kind == 'gain':
            raise ValueError('Kind must be either "gain" or "dB"!')
        return Signal(envs, samplerate=self.samplerate)

    def _apply_envelope(self, envelope, times, kind):
        # TODO: write tests for the new options!
        if hasattr(envelope, 'data') and hasattr(envelope, 'samplerate'): # it's a child object
            envelope = envelope.resample(samplerate=self.samplerate)
        elif isinstance(envelope, (list, numpy.ndarray)):  # make it a Signal, so we can use .n_samples and .n_channels
            envelope = Signal(numpy.array(envelope), samplerate=self.samplerate)
        new = copy.deepcopy(self)
        if times is None:
            times = numpy.linspace(0, 1, envelope.n_samples) * self.duration
        else:  # times vector was supplied
            if len(times) != envelope.n_samples:
                raise ValueError('Envelope and times need to be of equal length!')
            times = numpy.array(times)
            times[times > self.duration] = self.duration  # clamp between 0 and sound duration
            times[times < 0] = 0
        envelope_interp = numpy.empty((self.n_samples,envelope.n_channels)) # prep array for interpolated env
        for i in range(envelope.n_channels): # interp each channel
            envelope_interp[:,i] = numpy.interp(self.times, times, envelope[:,i])
        if kind == 'dB':
            envelope_interp = 10**(envelope_interp/20.) # convert dB to gain factors
        if envelope.n_channels == new.n_channels: # corresponding chans -> just multiply
            new.data *= envelope_interp
        elif envelope.n_channels == 1 and new.n_channels > 1: # env 1 chan, sound multichan -> apply env to each chan
            for i in range(new.n_channels):
                new.data[:,i] *= envelope_interp[:,0]
        else: # or neither (raise error)
            raise ValueError('Envelope needs to be 1d or have the same number of channels as the sound!')
        return new

    def delay(self, duration=1, channel=0, filter_length=2048):
        """
        Add a delay to one channel.

        Arguments:
            duration (int | float | array-like): duration of the delay in seconds (given a float) or samples (given
                an int). Given an array with the same length as the sound, each sample is delayed by the
                corresponding number of seconds. This option is used by in `slab.Binaural.itd_ramp`.
            channel (int): The index of the channel to add the delay to
            filter_length (int): Must be and even number. determines the accuracy of the reconstruction when
                using fractional sample delays. Defaults to 2048, or the sound length for shorter signals.
        Returns:
            (slab.Signal): a copy of the instance with the specified delay.
        """
        new = copy.deepcopy(self)
        if channel >= self.n_channels:
            raise ValueError('Channel must be smaller than number of channels in sound!')
        if filter_length % 2:
            raise ValueError('Filter_length must be even!')
        if self.n_samples < filter_length:  # reduce the filter_length to the sound length of short signals
            filter_length = self.n_samples - 1 if self.n_samples % 2 else self.n_samples  # make even
        center_tap = int(filter_length / 2)
        t = numpy.array(range(filter_length))
        if isinstance(duration, (int, float, numpy.int64, numpy.float64)):  # just a constant delay
            duration = Signal.in_samples(duration, self.samplerate)
            if duration > self.n_samples:
                raise ValueError("Duration of the delay cant be greater longer then the sound!")
            x = t - duration + 1
            window = 0.54 - 0.46 * numpy.cos(2 * numpy.pi * (x+0.5) /
                                             filter_length)  # Hamming window
            if numpy.abs(duration) < 1e-10:
                tap_weight = numpy.zeros_like(t)
                tap_weight[center_tap] = 1
            else:
                tap_weight = window * numpy.sinc(x-center_tap)
            new.data[:, channel] = numpy.convolve(self.data[:, channel], tap_weight, mode='same')
        else:  # dynamic delay
            if len(duration) != self.n_samples:
                ValueError('Duration shorter or longer than sound!')
            duration *= self.samplerate  # assuming vector in seconds, convert to samples
            padding = numpy.zeros(center_tap)
            # for zero-padded convolution (potential edge artifacts!)
            sig = numpy.concatenate((padding, new.channel(channel), padding), axis=None)
            for i, current_delay in enumerate(duration):
                x = t-current_delay
                window = 0.54 - 0.46 * numpy.cos(2 * numpy.pi *
                                                 (x+0.5) / filter_length)  # Hamming window
                if numpy.abs(current_delay) < 1e-10:
                    tap_weight = numpy.zeros_like(t)
                    tap_weight[center_tap] = 1
                else:
                    tap_weight = window * numpy.sinc(x-center_tap)
                    sig_portion = sig[i:i+filter_length]
                    # sig_portion and tap_weight have the same length, so the valid part of the convolution is just
                    # one sample, which gets written into the sound at the current index
                    new.data[i, channel] = numpy.convolve(sig_portion, tap_weight, mode='valid')[0]
        return new

    def plot_samples(self, show=True, axis=None):
        """
        Stem plot of the samples of the signal.

        Arguments:
            show (bool): whether to show the plot right after drawing.
            axis (matplotlib.axes.Axes | None): axis to plot to. If None create a new plot.
        """
        if matplotlib is False:
            raise ImportError('Plotting signals requires matplotlib.')
        if axis is None:
            _, axis = plt.subplots()
        if self.n_channels == 1:
            axis.stem(self.channel(0))
        else:
            for i in range(self.n_channels):
                axis.stem(self.channel(i), label=f'channel {i}', linefmt=f'C{i}')
            plt.legend()
        axis.set(title='Samples', xlabel='Number', ylabel='Value')
        if show:
            plt.show()
