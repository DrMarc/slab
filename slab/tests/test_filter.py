import slab
import numpy
import scipy


def test_cutoff():

    sound = slab.Sound.whitenoise()
    bandpass = slab.Filter.cutoff_filter(frequency=(500, 1000), kind='bp')
    bandpass.tf(show=False)
    sound = bandpass.apply(sound)


def test_equalization():

    sound = slab.Sound.pinknoise(samplerate=44100)
    filt = scipy.signal.firwin2(1000, freq=[0, 215, 1480, 5000, 12300, 22050],
                                gain=[1., 0.2, 1.1, 0.5, 1.0, 0.0], fs=sound.samplerate)
    recording = slab.Sound(scipy.signal.filtfilt(filt, 1, sound.data.flatten()), samplerate=44100)
    fbank = slab.Filter.equalizing_filterbank(sound, recording, low_cutoff=200, high_cutoff=16000)
    fbank.save('/tmp/equalizing_filter.npy')
    fbank = slab.Filter.load('/tmp/equalizing_filter.npy')
    sound_filt = fbank.apply(sound)
    Z_filt, _ = sound_filt.spectrum(show=False)
    Z_sound, _ = sound.spectrum(show=False)
    Z_rec, _ = recording.spectrum(show=False)
    # The difference between spectra should be smaller after equalization
    assert numpy.abs(Z_sound-Z_filt).sum() < numpy.abs(Z_sound-Z_rec).sum()
