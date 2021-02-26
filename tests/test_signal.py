import slab
import numpy
import unittest


class TestSoundMethods(unittest.TestCase):

    def test_signal_generation(self):
        # (numpy.ndarray | object | list)
        for i in range(100):
            n_samples = numpy.random.randint(100, 10000)
            n_channels = numpy.random.randint(1, 10)
            samplerate = numpy.random.randint(10, 1000)
            sig = slab.Signal(numpy.random.randn(n_samples), samplerate)
            self.assertEqual(sig.n_channels, n_channels)
            self.assertEqual(sig.n_samples, n_samples)
            self.assertEqual(sig.samplerate, samplerate)
            self.assertAlmostEqual(sig.times.max()*samplerate, n_samples, places=-1)
            self.assertEqual(len(sig.times), len(sig.data))

    def test_arithmetics(self):
        for _ in range(100):
            n_samples = numpy.random.randint(100, 10000)
            n_channels = numpy.random.randint(1, 10)
            samplerate = numpy.random.randint(10, 1000)
            sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
            numpy.testing.assert_equal((sig+sig).data, (sig*2).data)
            numpy.testing.assert_equal((sig-sig).data, numpy.zeros([n_samples, n_channels]))
            numpy.testing.assert_equal((sig*sig).data, sig**2)

    def test_samplerate(self):
        dur_seconds = numpy.abs(numpy.random.randn(100))
        dur_samples = numpy.random.randint(10, 100000, 100)
        for i in range(100):
            samplerate = numpy.random.randint(1, 100000)
            self.assertEqual(slab.Signal.in_samples(dur_seconds[i], samplerate), int(dur_seconds[i]*samplerate))
            self.assertEqual(slab.Signal.in_samples(dur_samples[i], samplerate), dur_samples[i])

    def test_channels(self):
        for _ in range(100):
            n_samples = numpy.random.randint(100, 10000)
            n_channels = numpy.random.randint(1, 10)
            samplerate = numpy.random.randint(10, 1000)
            sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
            for i, ch in enumerate(sig.channels()):
                numpy.testing.assert_equal(sig.channel(i).data, ch.data)

    def test_resize(self):
        dur_seconds = numpy.abs(numpy.random.randn(100))
        dur_samples = numpy.random.randint(10, 100000, 100)
        for i in range(100):
            n_samples = numpy.random.randint(100, 10000)
            n_channels = numpy.random.randint(1, 10)
            samplerate = numpy.random.randint(10, 1000)
            sig = slab.Signal(numpy.random.randn(n_samples, n_channels), samplerate=samplerate)
            resized = sig.resize(dur_seconds[i])
            self.assertAlmostEqual(resized.n_samples, dur_seconds[i]*resized.samplerate, places=0)
            resized = sig.resize(dur_samples[i])
            self.assertEqual(resized.n_samples, dur_samples[i])

    def test_resample(self):
        for i in range(100):
            sig = slab.Signal(numpy.random.randn(numpy.random.randint(100, 10000)))
            samplerate = numpy.random.randint(10, 1000)
            sig_resampled = sig.resample(samplerate)
            self.assertAlmostEqual(sig_resampled.n_samples, sig.duration*samplerate, places=0)

    def test_envelope(self):
        sig = slab.Sound.tone()
        _ = sig.get_envelope(kind="gain")  # TODO: check that the Hilbert envelope works properly
        _ = sig.get_envelope(kind="dB")
        for i in range(100):
            env = numpy.array([numpy.random.randn(), numpy.random.randn(), numpy.random.randn()])
            sig = sig.apply_envelope(envelope=env)
            self.assertEqual(sig.data.max(), numpy.abs(env).max())

    def test_delay(self):
        dur_seconds = numpy.abs(numpy.random.randn(100))
        dur_samples = numpy.random.randint(10, 100000, 100)
        for i in range(100):
            sig = slab.Signal(numpy.random.randn(numpy.random.randint(100, 10000), numpy.random.randint(1, 10)))
            channel = numpy.random.choice(sig.n_channels)
            for dur in [dur_samples[i], dur_seconds[i]]:
                delay_n_samples = slab.Signal.in_samples(dur, slab.Signal.get_samplerate(None))
            if delay_n_samples > sig.n_samples:
                self.assertRaises(ValueError, sig.delay, dur, channel)
            else:
                delayed = sig.delay(duration=dur, channel=channel, filter_length=numpy.random.randint(100, 10000)*2)
                delayed_channel = delayed.channel(channel)
                numpy.testing.assert_almost_equal(delayed_channel[:delay_n_samples].flatten(),
                                                  numpy.zeros(delay_n_samples), decimal=7)
