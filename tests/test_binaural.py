import slab
import numpy


def test_itd():
    for _ in range(100):
        sound = slab.Binaural.whitenoise()
        itd = numpy.random.uniform(-0.1, 0.1)
        lateral = sound.apply_itd(itd)
        itd = slab.Sound.in_samples(itd, 8000)
        assert itd == lateral.get_itd(max_lag=numpy.abs(itd))


def test_ild():
    sound = slab.Binaural.whitenoise()
    for _ in range(100):
        ild = numpy.random.uniform(-10, 10)
        lateral = sound.apply_ild(ild)
        numpy.testing.assert_almost_equal(numpy.diff(lateral.level)*-1, ild, decimal=5)

def test_azimuth_to_itd_ild():
    for _ in range(100):
        frequency = numpy.random.randint(50, 2000)
        headsize = numpy.random.uniform(7, 11)
        itds = []
        ilds = []
        for azimuth in numpy.linspace(-90, 0, 20):
            itds.append(slab.Binaural.azimuth_to_itd(azimuth, frequency, headsize))
            ilds.append(slab.Binaural.azimuth_to_ild(azimuth, frequency))
        assert all([itds[i] < itds[i+1] for i in range(len(itds)-1)])
        itds = []
        for azimuth in numpy.linspace(90, 0, 20):
            itds.append(slab.Binaural.azimuth_to_itd(azimuth, frequency, headsize))
            ilds.append(slab.Binaural.azimuth_to_ild(azimuth, frequency))
            assert all([itds[i] > itds[i + 1] for i in range(len(itds) - 1)])


def test_at_azimuth():
    for _ in range(10):
        sound = slab.Binaural.whitenoise()
        for azimuth in numpy.linspace(-90, 90, 40):
            lateral = sound.at_azimuth(azimuth)
            itd = slab.Sound.in_samples(slab.Binaural.azimuth_to_itd(azimuth), 8000)
            assert numpy.abs(itd - lateral.get_itd()) <= 1
            ild = slab.Binaural.azimuth_to_ild(azimuth)
            numpy.testing.assert_almost_equal(ild, numpy.diff(lateral.level)*-1, decimal=0)
