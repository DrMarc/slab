import slab
import numpy
from copy import deepcopy


def test_itd():
    for _ in range(100):
        sound = slab.Binaural.whitenoise()
        itd = numpy.random.uniform(-0.1, 0.1)
        lateral = sound.itd(duration=itd)
        itd = slab.Sound.in_samples(itd, 8000)
        assert itd == lateral.itd(max_lag=numpy.abs(itd))


def test_ild():
    sound = slab.Binaural.whitenoise()
    for _ in range(100):
        ild = numpy.random.uniform(-10, 10)
        lateral = sound.ild(ild)
        numpy.testing.assert_almost_equal(lateral.ild(), ild, decimal=5)


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


def test_itd_ramp():
    for _ in range(10):
        sound = slab.Binaural.whitenoise()
        from_itd = numpy.random.uniform(-0.001, 0.)
        to_itd = numpy.random.uniform(0.001, 0.)
        moving = sound.itd_ramp(from_itd=from_itd, to_itd=to_itd)
        start, stop = deepcopy(moving), deepcopy(moving)
        start.data = start.data[0:200, :]
        stop.data = stop.data[-201:-1, :]
        assert numpy.abs(start.itd() - slab.Sound.in_samples(from_itd, 8000)) <= 1
        assert numpy.abs(stop.itd() - slab.Sound.in_samples(to_itd, 8000)) <= 1


def test_ild_ramp():
    for _ in range(100):
        sound = slab.Binaural.whitenoise()
        from_ild = numpy.random.uniform(-10, 0.)
        to_ild = numpy.random.uniform(10, 0.)
        moving = sound.ild_ramp(from_ild=from_ild, to_ild=to_ild)
        start, stop = deepcopy(moving), deepcopy(moving)
        start.data = start.data[0:200, :]
        stop.data = stop.data[-201:-1, :]
        numpy.testing.assert_almost_equal(numpy.diff(start.level), from_ild, decimal=0)
        numpy.testing.assert_almost_equal(numpy.diff(stop.level), to_ild, decimal=0)


def test_externalize():
    sound = slab.Binaural.whitenoise()
    external = sound.externalize()

