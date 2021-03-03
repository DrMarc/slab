import slab
import numpy
import pathlib
import tempfile

tmpdir = pathlib.Path(tempfile.gettempdir())


def test_sound_generation():
    # numpy.ndarray | str | pathlib.Path | list
    for _ in range(100):
        data = numpy.ones([10, 2])
        sound = slab.Sound(data, samplerate=10)  # generate sound from array
        sound1 = slab.Sound([data, data], samplerate=10)  # generate sound from list of arrays
        sound2 = slab.Sound([sound, sound])  # generate from list of sounds
        # sound1 and sound 2 should be the same:
        numpy.testing.assert_equal(sound1.data, sound2.data)
        numpy.testing.assert_equal(sound1.times, sound2.times)
        assert sound1.samplerate == sound2.samplerate
        assert sound1.duration == sound2.duration
        assert sound1.duration == sound2.duration
        # test if saving the file and initializing from string / path works. The reading and writing of data
        # is tested in more detail in test_read_write()
        sound = slab.Sound(numpy.random.randn(1000, 2), samplerate=numpy.random.randint(100,1000))
        sound.write(tmpdir/"sound.wav", normalise=False)
        loaded1 = slab.Sound(tmpdir/"sound.wav")
        loaded2 = slab.Sound(str(tmpdir/"sound.wav"))
        numpy.testing.assert_equal(loaded1.data, loaded2.data)
        numpy.testing.assert_equal(loaded1.times, loaded2.times)


def test_read_write():
    pass


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


def test_frames():
    for _ in range(100):
        frame_dur = numpy.random.randint(1, 5000)
        step_size = numpy.floor(frame_dur / numpy.sqrt(numpy.pi) / 8).astype(int)
        sound_dur = numpy.abs(numpy.random.randn())
        sound = slab.Sound.whitenoise(duration=sound_dur)
        window_centers = sound.frametimes(duration=frame_dur)
        windows = sound.frames(duration=frame_dur)
        for window, center in zip(windows, window_centers):
            center1 = window[frame_dur][0]
            center2 = sound[numpy.where(sound.times == center)[0][0]][0]
            numpy.testing.assert_almost_equal(center1, center2, decimal=3)


