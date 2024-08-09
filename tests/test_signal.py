import numpy
import pytest
import slab

def test_signal_generation():
    # (numpy.ndarray | object | list)
    for _ in range(100):
        n_samples = numpy.random.randint(100, 10000)
        n_channels = numpy.random.randint(1, 10)
        samplerate = numpy.random.randint(10, 1000)
        sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate)
        assert sig.n_channels == n_channels
        assert sig.n_samples == n_samples
        assert sig.samplerate == samplerate
        assert len(sig.times) == len(sig.data)
        numpy.testing.assert_almost_equal(sig.times.max()*samplerate, n_samples, decimal=-1)


def test_arithmetics():
    for _ in range(100):
        n_samples = numpy.random.randint(100, 10000)
        n_channels = numpy.random.randint(1, 10)
        samplerate = numpy.random.randint(10, 1000)
        sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
        numpy.testing.assert_equal((sig+sig).data, (sig*2).data)
        numpy.testing.assert_equal((sig-sig).data, numpy.zeros([n_samples, n_channels]))
        numpy.testing.assert_equal((sig*sig).data, sig**2)


def test_samplerate():
    samples = slab.Signal.in_samples([1.,2], 1000)
    assert samples[0] == 1000
    assert samples[1] == 2
    orig_rate = slab.signal._default_samplerate
    slab.set_default_samplerate(10000)
    assert slab.signal._default_samplerate == 10000
    slab.set_default_samplerate(orig_rate)
    dur_seconds = numpy.abs(numpy.random.randn(100))
    dur_samples = numpy.random.randint(10, 100000, 100)
    for i in range(100):
        samplerate = numpy.random.randint(1, 100000)
        assert numpy.abs(slab.Signal.in_samples(dur_seconds[i], samplerate) - int(dur_seconds[i]*samplerate)) <= 1
        assert slab.Signal.in_samples(dur_samples[i], samplerate) == dur_samples[i]


def test_channels():
    for _ in range(100):
        n_samples = numpy.random.randint(100, 10000)
        n_channels = numpy.random.randint(1, 10)
        samplerate = numpy.random.randint(10, 1000)
        sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
        for i, ch in enumerate(sig.channels()):
            numpy.testing.assert_equal(sig.channel(i).data, ch.data)


def test_resize():
    dur_seconds = numpy.abs(numpy.random.randn(100))
    dur_samples = numpy.random.randint(10, 100000, 100)
    for i in range(100):
        n_samples = numpy.random.randint(100, 10000)
        n_channels = numpy.random.randint(1, 10)
        samplerate = numpy.random.randint(10, 1000)
        sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
        resized = sig.resize(dur_seconds[i])
        assert numpy.abs(resized.n_samples - dur_seconds[i]*resized.samplerate) < 1
        resized = sig.resize(dur_samples[i])
        assert resized.n_samples == dur_samples[i]


def test_trim():
    for _ in range(100):
        n_samples = numpy.random.randint(100, 10000)
        n_channels = numpy.random.randint(1, 10)
        samplerate = numpy.random.randint(10, 1000)
        sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
        start, stop = numpy.sort(numpy.random.randint(0, n_samples+1, 2))
        if start == stop:
            start -= 1
        if numpy.random.rand() < 0.5:
            trimmed = sig.trim(start=start, stop=stop)
        else:
            trimmed = sig.trim(start=start/samplerate, stop=stop/samplerate)
        assert numpy.abs(trimmed.n_samples - (stop-start)) <= 1
    with pytest.raises(ValueError):  # testing start not preceding stop case
        trimmed = sig.trim(start=stop, stop=start)


def test_resample():
    for _ in range(100):
        sig = slab.Signal(numpy.random.randn(numpy.random.randint(100, 10000)))
        samplerate = numpy.random.randint(10, 1000)
        sig_resampled = sig.resample(samplerate)
        assert numpy.abs(sig_resampled.n_samples - sig.duration*samplerate) < 1


def test_envelope():
    sig = slab.Sound.tone()
    _ = sig.envelope(kind="gain")
    _ = sig.envelope(kind="dB")
    _ = sig.envelope(apply_envelope=[0, 1, 0], times=[0, 0.3, 1])
    for _ in range(100):
        env = numpy.random.uniform(-1, 1, 3)
        sig2 = sig.envelope(apply_envelope=env)
        assert numpy.abs(sig2.data.max() - sig.data.max() * numpy.abs(env).max()) < .001


def test_delay():
    dur_seconds = numpy.abs(numpy.random.randn(100))
    dur_samples = numpy.random.randint(10, 100000, 100)
    for i in range(100):
        sig = slab.Signal(numpy.random.randn(numpy.random.randint(100, 10000), numpy.random.randint(1, 10)))
        channel = numpy.random.choice(sig.n_channels)
        for dur in [dur_samples[i], dur_seconds[i]]:
            delay_n_samples = slab.Signal.in_samples(dur, slab.signal._default_samplerate)
        if delay_n_samples > sig.n_samples:
            numpy.testing.assert_raises(ValueError, sig.delay, dur, channel)
        else:
            delayed = sig.delay(duration=dur, channel=channel, filter_length=numpy.random.randint(100, 10000)*2)
            delayed_channel = delayed.channel(channel)
            numpy.testing.assert_almost_equal(delayed_channel[:delay_n_samples].flatten(),
                                              numpy.zeros(delay_n_samples), decimal=7)
