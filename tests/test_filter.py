import itertools
import slab
import numpy
import scipy


def test_low_high_pass():
    for i in range(10):
        sound = slab.Sound.whitenoise(duration=2.0)
        for kind, fir in itertools.product(["lp", "hp"], [False, True]):
            edge_freq = numpy.random.uniform(100, 2000)
            length = numpy.random.randint(500, 5000)
            filt = slab.Filter.band(frequency=edge_freq, length=length, kind=kind, fir=fir)
            filt_sound = filt.apply(sound)
            Z, freqs = filt_sound.spectrum(show=False)
            idx = numpy.abs(freqs-edge_freq).argmin()
            if kind == "hp":
                suppressed = Z[0:idx]
            else:
                suppressed = Z[idx:]
            assert suppressed.max() < -40


def test_band_pass_stop():
    sound = slab.Sound.whitenoise(duration=2.0)
    for kind, fir in itertools.product(["bp", "bs"], [False, True]):
        lower_edge_freq = numpy.random.uniform(100, 1000)
        higher_edge_freq = lower_edge_freq + numpy.random.uniform(100, 1000)
        length = numpy.random.randint(500, 5000)
        filt = slab.Filter.band(frequency=(lower_edge_freq, higher_edge_freq), length=length, kind=kind, fir=fir)
        filt_sound = filt.apply(sound)
        Z, freqs = filt_sound.spectrum(show=False)
        low_idx = numpy.abs(freqs - lower_edge_freq).argmin()
        high_idx = numpy.abs(freqs - higher_edge_freq).argmin()
        if kind == "bp":
            suppressed = numpy.concatenate([Z[0:low_idx], Z[high_idx:]])
        else:
            suppressed = Z[low_idx:high_idx]
        assert suppressed.max() < -40


def test_custom_band():
    sound = slab.Sound.whitenoise(duration=2.0, samplerate=44100)
    freqs = numpy.array([100., 800., 2000., 4300., 8000., 14500., 18000.])
    gains = [
        [0., 1., 0., 1., 0., 1., 0.],
        [1., 0., 1, 0., 1., 0., 1.],
        [0., 1., 0., 0., 1., 1., 0.],
        [1., 0., 1., 0., 0., 1., 0.]
    ]
    for i in range(10):
        for fir, gain in itertools.product([True, False], gains):
            freqs += numpy.random.uniform(1, 10, 7)
            freqs.sort()
            length = numpy.random.randint(500, 5000)
            filt = slab.Filter.band(frequency=list(freqs), gain=gain, length=length, fir=fir, samplerate=sound.samplerate)
            w, h = filt.tf(show=False)
            suppressed_freqs = freqs[numpy.where(numpy.array(gain) == 0.0)]
            idx = [numpy.abs(w-freq).argmin() for freq in suppressed_freqs]
            assert max(h[idx]) < -20


def test_equalization():

    sound = slab.Sound.pinknoise(samplerate=44100)
    filt = scipy.signal.firwin2(1000, freq=[0, 215, 1480, 5000, 12300, 22050],
                                gain=[1., 0.2, 1.1, 0.5, 1.0, 0.0], fs=sound.samplerate)
    recording = slab.Sound(scipy.signal.filtfilt(filt, 1, sound.data.flatten()), samplerate=44100)
    fbank = slab.Filter.equalizing_filterbank(sound, recording, low_cutoff=200, high_cutoff=16000)
    fbank.save('/tmp/equalizing_filter.npy')
    fbank = slab.Filter.load('/tmp/equalizing_filter.npy')
    sound_filt = fbank.apply(sound)
    Z_filt, _ = sound_filt.spectrum(show=False)
    Z_sound, _ = sound.spectrum(show=False)
    Z_rec, _ = recording.spectrum(show=False)
    # The difference between spectra should be smaller after equalization
    assert numpy.abs(Z_sound-Z_filt).sum() < numpy.abs(Z_sound-Z_rec).sum()
