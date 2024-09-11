import inspect
import slab
import numpy
from copy import deepcopy

ils = slab.Binaural.make_interaural_level_spectrum()
hrtf = slab.HRTF.kemar()

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
            ilds.append(slab.Binaural.azimuth_to_ild(azimuth, frequency, ils))
        assert all([itds[i] < itds[i+1] for i in range(len(itds)-1)])
        itds = []
        for azimuth in numpy.linspace(90, 0, 20):
            itds.append(slab.Binaural.azimuth_to_itd(azimuth, frequency, headsize))
            ilds.append(slab.Binaural.azimuth_to_ild(azimuth, frequency, ils))
            assert all([itds[i] > itds[i + 1] for i in range(len(itds) - 1)])


def test_at_azimuth():
    for _ in range(10):
        sound = slab.Binaural.whitenoise()
        for azimuth in numpy.linspace(-90, 90, 40):
            lateral = sound.at_azimuth(azimuth, ils=ils)
            itd = slab.Sound.in_samples(slab.Binaural.azimuth_to_itd(azimuth), 8000)
            assert numpy.abs(itd - lateral.itd()) <= 1
            ild = slab.Binaural.azimuth_to_ild(azimuth, ils=ils)
            assert numpy.abs(ild - lateral.ild()) <= 3


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
    for _ in range(10):
        idx_frontal = numpy.where((hrtf.sources.vertical_polar[:, 1] == 0) &
                                  (hrtf.sources.vertical_polar[:, 0] == 0))[0][0]
        sound = slab.Binaural.whitenoise(samplerate=hrtf.samplerate)
        filtered = hrtf.data[idx_frontal].apply(sound)
        filtered = filtered.resize(sound.duration)
        external = sound.externalize()
        assert numpy.abs(filtered.data-external.data).sum() < numpy.abs(filtered.data-sound.data).sum()
        assert numpy.abs(sound.level - external.level).max() < 0.6


def test_interaural_level_spectrum():
    sound = slab.Binaural.whitenoise(samplerate=int(ils['samplerate']))
    azimuths = ils['azimuths']
    for i, azi in enumerate(azimuths):
        level_differences = ils['level_diffs'][:, i]
        lateral = sound.interaural_level_spectrum(azi, ils)
        fbank = slab.Filter.cos_filterbank(samplerate=sound.samplerate, pass_bands=True)
        subbands_left = fbank.apply(lateral.left)
        subbands_right = fbank.apply(lateral.right)
        assert -1 < (level_differences - (subbands_left.level - subbands_right.level)).mean() < 1


def test_overloaded_sound_generators():
    methods = ['chirp', 'click', 'clicktrain', 'dynamic_tone', 'equally_masking_noise',
               'harmoniccomplex', 'irn', 'multitone_masker', 'pinknoise', 'powerlawnoise',
               'silence', 'tone', 'vowel', 'whitenoise']
    for method in methods:
        func = getattr(slab.Binaural, method)
        assert func().n_channels == 2
        assert func().name != 'unnamed'


def test_drr():
    for _ in range(10):
        steepness = numpy.random.randint(1, 3)
        duration = numpy.random.uniform(2.0, 8.0)
        decay_resolution = 10
        decay_curve = [numpy.exp(-i * steepness) for i in range(decay_resolution)]
        sound = slab.Binaural.whitenoise(kind='dichotic', duration=duration, samplerate=44100)
        impulse = sound.envelope(apply_envelope=decay_curve)
        winlength = numpy.random.uniform(0.0001, 0.01)
        if numpy.random.randint(0, 2):
            winlength = max(2, impulse.in_samples(winlength, impulse.samplerate))
        if numpy.random.randint(0, 2):
            drr = impulse.drr(winlength=winlength)
        else:
            drr = impulse.drr()
        assert 0 > drr > -100
        assert isinstance(drr, float)
