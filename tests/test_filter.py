import itertools
import pathlib
import tempfile
import slab
import numpy
tmpdir = pathlib.Path(tempfile.gettempdir())

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
            assert suppressed.max() < -25


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
        assert suppressed.max() < -30


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


def test_cos_filterbank():
    for i in range(10):
        sound = slab.Sound.whitenoise(duration=1.0, samplerate=44100)
        length = numpy.random.randint(1000, 5000)
        low_cutoff = numpy.random.randint(0, 500)
        high_cutoff = numpy.random.choice([numpy.random.randint(5000, 15000), None])
        pass_bands = False
        n_filters = []
        for bandwidth in numpy.linspace(0.1, 0.9, 9):
            fbank = slab.Filter.cos_filterbank(length, bandwidth, low_cutoff, high_cutoff, pass_bands, sound.samplerate)
            n_filters.append(fbank.n_filters)
            filtsound = fbank.apply(sound)
            assert filtsound.n_channels == fbank.n_filters
            assert filtsound.n_samples == sound.n_samples
        assert all([n_filters[i] >= n_filters[i+1] for i in range(len(n_filters)-1)])
        bandwidth = numpy.random.uniform(0.1, 0.9)
        pass_bands = True
        fbank = slab.Filter.cos_filterbank(sound.n_samples, bandwidth, low_cutoff, high_cutoff, pass_bands,
                                           sound.samplerate)
        filtsound = fbank.apply(sound)
        collapsed = slab.Filter.collapse_subbands(filtsound, fbank)
        numpy.testing.assert_almost_equal(sound.data, collapsed.data, decimal=-1)


def test_center_freqs():
    for i in range(100):
        low_cutoff = numpy.random.randint(0, 500)
        high_cutoff = numpy.random.choice([numpy.random.randint(5000, 20000)])
        bandwidth1 = numpy.random.uniform(0.1, 0.7)
        pass_bands = False
        center_freqs1, bandwidth2, _ = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth1, pass_bands)
        assert numpy.abs(bandwidth1 - bandwidth2) < 0.3
        fbank = slab.Filter.cos_filterbank(5000, bandwidth1, low_cutoff, high_cutoff, pass_bands, 44100)
        center_freqs2 = fbank.filter_bank_center_freqs()
        assert numpy.abs(slab.Filter._erb2freq(center_freqs1[1:]) - center_freqs2[1:]).max() < 40
        assert numpy.abs(center_freqs1 - slab.Filter._freq2erb(center_freqs2)).max() < 1


def test_equalization():
    for i in range(10):
        length = numpy.random.randint(1000, 5000)
        low_cutoff = numpy.random.randint(20, 2000)
        high_cutoff = numpy.random.randint(10000, 20000)
        sound = slab.Sound.pinknoise(samplerate=44100)
        filt = slab.Filter.band(frequency=[100., 800., 2000., 4300., 8000., 14500., 18000.],
                                gain=[0., 1., 0., 1., 0., 1., 0.], samplerate=sound.samplerate)
        filtered = filt.apply(sound)
        fbank = slab.Filter.equalizing_filterbank(sound, filtered, low_cutoff=200, high_cutoff=16000)
        equalized = fbank.apply(sound)
        Z_equalized, _ = equalized.spectrum(show=False)
        Z_sound, _ = sound.spectrum(show=False)
        Z_filtered, _ = filtered.spectrum(show=False)
        # The difference between spectra should be smaller after equalization
        assert numpy.abs(Z_sound-Z_filtered).sum() / numpy.abs(Z_sound-Z_equalized).sum() > 2


def test_load_save():
    for kind, freq in zip(["lp", "hp", "bs", "bp"], [
        numpy.random.uniform(100, 2000),
        numpy.random.uniform(100, 2000),
        (0+numpy.random.uniform(100, 2000), 2000+numpy.random.uniform(100, 2000)),
        (0 + numpy.random.uniform(100, 2000), 2000 + numpy.random.uniform(100, 2000))
    ]):
        for fir in (True, False):
            filt = slab.Filter.band(kind=kind, frequency=freq, fir=fir)
            filt.save(tmpdir/"filt.npy")
            loaded = slab.Filter.load(tmpdir/"filt.npy")
            numpy.testing.assert_equal(filt.data, loaded.data)
            numpy.testing.assert_equal(filt.times, loaded.times)
            assert filt.fir == loaded.fir
            assert filt.n_frequencies == loaded.n_frequencies
            assert filt.n_taps == loaded.n_taps


