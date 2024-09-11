import pathlib
import tempfile
import numpy
import scipy.signal
import slab
tmpdir = pathlib.Path(tempfile.gettempdir())


def test_sound_generation():
    # numpy.ndarray | str | pathlib.Path | list
    for _ in range(20):
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
        sound = slab.Sound(numpy.random.randn(1000, 2), samplerate=numpy.random.randint(100, 1000))
        sound.write(tmpdir/"sound.wav", normalise=False)
        loaded1 = slab.Sound(tmpdir/"sound.wav")
        loaded2 = slab.Sound(str(tmpdir/"sound.wav"))
        numpy.testing.assert_equal(loaded1.data, loaded2.data)
        numpy.testing.assert_equal(loaded1.times, loaded2.times)
    # test if .name attribute of loaded sound object is the path string
    assert loaded1.name == f"{tmpdir / 'sound.wav'}"

def test_read_write():
    for _ in range(20):
        for normalize in [True, False]:
            sound = slab.Sound(numpy.random.randn(1000, 2), samplerate=numpy.random.randint(100, 1000))
            if normalize is False:
                sound.data = sound.data / sound.data.max()
            sound.write(tmpdir / "sound.wav", normalise=True)
            loaded = slab.Sound(tmpdir/"sound.wav")
            loaded.level = sound.level
            numpy.testing.assert_almost_equal(sound.data, loaded.data, decimal=3)


def test_tone():
    for freq in range(50, 20000, 500):
        sound = slab.Sound.tone(duration=numpy.random.randint(1000, 5000), frequency=freq, samplerate=44100)
        Z, freqs = sound.spectrum(show=False)
        assert numpy.abs(freqs[numpy.where(Z == Z.max())[0][0]] - freq) < 50
        assert sound.name == f'tone_{str(freq)}'
    for freq in range(500, 5000, 200):
        harmonic = slab.Sound.harmoniccomplex(duration=numpy.random.randint(1000, 5000), f0=freq, samplerate=44100)
        Z, freqs = harmonic.spectrum(show=False)
        peaks = scipy.signal.find_peaks(Z.flatten())[0]
        peak_freqs = freqs[peaks]
        peak_freqs = peak_freqs/freq
        numpy.testing.assert_almost_equal(peak_freqs, numpy.linspace(1, len(peaks), len(peaks)), decimal=0)
        assert harmonic.name == f'harmonic_{str(freq)}'


def test_powerlawnoise():
    for _ in range(20):
        centroids = []
        for alpha in numpy.linspace(.5, 1., 3):
            sound = slab.Sound.powerlawnoise(alpha=alpha, samplerate=44100)
            assert sound.name == f'powerlawnoise_{str(alpha)}'
            centroids.append(sound.spectral_feature("centroid"))
        assert all([centroids[i] > centroids[i+1] for i in range(len(centroids)-1)])


def test_crossfade():
    import itertools
    samplerate = 44100
    noise_durations = [0.1, 0.5, 1.0]
    vowel_durations = [0.1, 0.5, 1.0]
    overlaps = [0.0, 0.01]
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
    noise_x_noise = slab.Sound.crossfade(noise, noise)
    assert noise_x_noise.name == f'{noise.name}_x_{noise.name}'


def test_frames():
    for _ in range(20):
        frame_dur = numpy.random.randint(10, 5000)
        sound_dur = numpy.abs(numpy.random.randn())+0.1
        sound = slab.Sound.whitenoise(duration=sound_dur)
        window_centers = sound.frametimes(duration=frame_dur)
        windows = sound.frames(duration=frame_dur)
        for window, center in zip(windows, window_centers):
            center1 = window[frame_dur][0]
            center2 = sound[numpy.where(sound.times == center)[0][0]][0]
            numpy.testing.assert_almost_equal(center1, center2, decimal=1)
