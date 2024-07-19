import itertools
import pathlib
import tempfile
import slab
import numpy
tmpdir = pathlib.Path(tempfile.gettempdir())

def test_low_high_pass():
    for i in range(10):
        sound = slab.Sound.whitenoise(duration=2.0)
        for kind, fir in itertools.product(["lp", "hp"], ['FIR', 'IR', 'TF']):
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
    for kind, fir in itertools.product(["bp", "bs"], ['FIR', 'IR', 'TF']):
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
        for fir, gain in itertools.product(['FIR', 'IR', 'TF'], gains):
            freqs += numpy.random.uniform(1, 10, 7)
            freqs.sort()
            length = numpy.random.randint(500, 5000)
            filt = slab.Filter.band(frequency=list(freqs), gain=gain, length=length, fir=fir, samplerate=sound.samplerate)
            w, h = filt.tf(show=False)
            suppressed_freqs = freqs[numpy.where(numpy.array(gain) == 0.0)]
            idx = [numpy.abs(w-freq).argmin() for freq in suppressed_freqs]
            assert max(h[idx]) < -20


def test_cos_filterbank():
    for _ in range(10):
        sound = slab.Sound.whitenoise(duration=1.0, samplerate=44100)
        length = numpy.random.randint(1000, 5000)
        low_cutoff = numpy.random.randint(0, 500)
        high_cutoff = numpy.random.choice([numpy.random.randint(5000, 15000), None])
        n_filters = []
        for bandwidth in numpy.linspace(0.1, 0.9, 9):
            fbank = slab.Filter.cos_filterbank(length=length, bandwidth=bandwidth, low_cutoff=low_cutoff, high_cutoff=high_cutoff, pass_bands=False, samplerate=sound.samplerate)
            n_filters.append(fbank.n_filters)
            filtsound = fbank.apply(sound)
            assert filtsound.n_channels == fbank.n_filters
            assert filtsound.n_samples == sound.n_samples
        assert all([n_filters[i] >= n_filters[i+1] for i in range(len(n_filters)-1)])
        bandwidth = numpy.random.uniform(0.1, 0.9)
        fbank = slab.Filter.cos_filterbank(
                length=sound.n_samples, bandwidth=bandwidth, low_cutoff=low_cutoff,
                high_cutoff=high_cutoff, pass_bands=True, samplerate=sound.samplerate)
        filtsound = fbank.apply(sound)
        collapsed = slab.Filter.collapse_subbands(filtsound, fbank)
        numpy.testing.assert_almost_equal(sound.data, collapsed.data, decimal=-1)


def test_center_freqs():
    for _ in range(100):
        low_cutoff = numpy.random.randint(0, 500)
        high_cutoff = numpy.random.choice([numpy.random.randint(5000, 20000)])
        bandwidth = numpy.random.uniform(0.1, 0.7)
        center_freqs1, bandwidth1, erb_spacing1 = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth=bandwidth, pass_bands=False)
        center_freqs2, bandwidth2, erb_spacing2 = slab.Filter._center_freqs(low_cutoff, high_cutoff, bandwidth=bandwidth, pass_bands=True)
        assert all(center_freqs1 == center_freqs2[1:-1])
        assert len(center_freqs1) == len(center_freqs2)-2
        assert bandwidth1 == bandwidth2
        assert erb_spacing1 == erb_spacing2
        n_filters = len(center_freqs1)
        center_freqs3, bandwidth3, erb_spacing3 = slab.Filter._center_freqs(low_cutoff, high_cutoff, n_filters=n_filters, pass_bands=False)
        assert all(center_freqs1 == center_freqs3)
        assert bandwidth1 == bandwidth3
        assert erb_spacing1 == erb_spacing3


def test_equalization():
    for i in range(10):
        length = numpy.random.randint(1000, 5000)
        low_cutoff = numpy.random.randint(20, 2000)
        high_cutoff = numpy.random.randint(10000, 20000)
        sound = slab.Sound.pinknoise(samplerate=44100, duration=length/1000)
        # bandpass to reduce the difference at low and high frequencies
        sound = sound.filter([20, 18000], 'bp')
        freqs = numpy.logspace(numpy.log2(50), numpy.log2(18000), 10, base=2)
        gains = 2 ** numpy.random.uniform(-2, 2, freqs.shape)
        filt = slab.Filter.band(frequency=freqs.tolist(),
                                gain=gains.tolist(), samplerate=sound.samplerate)
        filtered = filt.apply(sound)
        fbank = slab.Filter.equalizing_filterbank(sound, filtered, low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        equalized = fbank.apply(filtered)
        Z_equalized, _ = equalized.spectrum(low_cutoff, high_cutoff, show=False)
        Z_sound, _ = sound.spectrum(low_cutoff, high_cutoff, show=False)
        Z_filtered, _ = filtered.spectrum(low_cutoff, high_cutoff, show=False)
        # The spectral difference between original and equalized should be minimum
        # TODO: now only check the average. might make sense to also check the maximum
        # TODO: the edge effect is annoying. should think of a way to deal with it
        assert numpy.abs(Z_sound-Z_equalized)[100:-1000].mean() < 1


def test_load_save():
    for kind, freq in zip(["lp", "hp", "bs", "bp"], [
        numpy.random.uniform(100, 2000),
        numpy.random.uniform(100, 2000),
        (0+numpy.random.uniform(100, 2000), 2000+numpy.random.uniform(100, 2000)),
        (0 + numpy.random.uniform(100, 2000), 2000 + numpy.random.uniform(100, 2000))
    ]):
        for fir in ('FIR', 'IR', 'TF'):
            filt = slab.Filter.band(kind=kind, frequency=freq, fir=fir)
            filt.save(tmpdir/"filt.npy")
            loaded = slab.Filter.load(tmpdir/"filt.npy")
            numpy.testing.assert_equal(filt.data, loaded.data)
            numpy.testing.assert_equal(filt.times, loaded.times)
            assert filt.fir == loaded.fir
            assert filt.n_frequencies == loaded.n_frequencies
            assert filt.n_taps == loaded.n_taps
