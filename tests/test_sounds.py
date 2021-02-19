import slab
import numpy


def test_properties():
    slab.calibrate(intensity=80, make_permanent=False)
    sound = slab.Sound(numpy.ones([10, 2]), samplerate=10)
    sound = sound.repeat(n=5)
    assert sound.samplerate == 10
    assert sound.n_samples == 50
    assert sound.duration == 5.0
    assert sound.n_channels == 2


def test_tone():
    sound = slab.Sound.multitone_masker()
    sound = slab.Sound.clicktrain()
    # sound = slab.Sound.dynamicripple() --> does not work
    sound = slab.Sound.chirp()
    sound = slab.Sound.tone()
    sound = slab.Sound.harmoniccomplex(f0=200, amplitude=[0, -10, -20, -30])
    sound.level = 80


def test_vowel():
    vowel = slab.Sound.vowel(vowel='a', duration=.5, samplerate=8000)
    vowel.ramp()
    vowel.spectrogram(dyn_range=50, show=False)
    vowel.spectrum(low=100, high=4000, log_power=True, show=False)
    vowel.waveform(start=0, end=.1, show=False)
    vowel.cochleagram(show=False)
    vowel.vocode()


def test_noise():
    sound = slab.Sound.erb_noise()
    sound = slab.Sound.powerlawnoise()
    sound = slab.Sound.irn()
    sound = slab.Sound.whitenoise(normalise=True)
    assert max(sound) <= 1
    assert min(sound) >= -1


def test_manipulations():
    sound1 = slab.Sound.pinknoise()
    sound2 = slab.Sound.pinknoise()
    sound1.pulse()
    sound1.am()
    sound2.aweight()
    sound = slab.Sound.crossfade(sound1, sound2, overlap=0.01)
    for feat in ['centroid', 'fwhm', 'flux', 'rolloff', 'flatness']:
        sound.spectral_feature(feature=feat)
    sound.crest_factor()
    sound.onset_slope()


def test_crossfade():
    import itertools
    samplerate = 44100
    noise_durations = [0.1, 0.5, 1.0]
    vowel_durations = [0.1, 0.5, 1.0]
    overlaps = [0.0, 0.1]
    combinations = itertools.product(noise_durations, vowel_durations, overlaps)
    for noise_dur, vowel_dur, overlap in combinations:
        noise = slab.Sound.whitenoise(duration=noise_dur, samplerate=samplerate)
        vowel = slab.Sound.vowel(duration=vowel_dur, samplerate=samplerate)
        expected_n_samples = int(noise.n_samples + vowel.n_samples*2 - ((samplerate*overlap)*2))
        noise2vowel = slab.Sound.crossfade(vowel, noise, vowel, overlap=overlap)
        assert noise2vowel.n_samples == expected_n_samples
        if overlap == 0:  # crossfade with overlap 0 should be the same as sequence
            noise2vowel_seq = slab.Sound.sequence(vowel, noise, vowel)
            assert all(noise2vowel.data == noise2vowel_seq.data)

