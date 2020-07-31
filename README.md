# slab

Class for working with sounds, including loading/saving, manipulating and playing \
install: `pip install git+https://github.com/DrMarc/soundtools.git`

* Signals
* Sound (inherits from Signals, methods for generating, manipulating, displaying, and analysing sound stimuli)
* Binaural (inherits from Sound, contains convenience functions for binaural stimuli)
* Filter (inherits from Signals, a stub at the moment)
* HRTF (inherits from Filter, quick interface to the .sofa format for spatial audio and functions for working with binaural filter functions)
* Psychoacoustics (a collection of classes for working trial sequences and psychoacoustic testing)


Much of the basic functionality and software architecture of the Signal and Sound classes is based on [brian.hears] (www.briansimulator.org/docs/hears.html). The classes are used primarily to lower the entrance barrier for working with sounds in Python for students in our lab, and provide easy access to typical operations in psychoacoustics.

**Examples**:
```python
>>> import slab
>>> sig = slab.Sound.tone()
>>> sig.level = 80
```

**Properties:**
```python
>>> sig.duration
1.0
>>> sig.nsamples
10
>>> sig.nchannels
2
```

**Slicing**
Signals implement __getitem__ and __setitem___ and supports slicing.
Slicing returns numpy.ndarrays or floats, not Signal objects.
You can also set values using slicing:
```python
>>> sig[:5] = 0
```
will set the first 5 samples to zero.
You can also select a subset of channels:
```python
>>> sig[:,1]
array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
```
would be data in the second channel. To extract a channel as a Signal or subclass object use `sig.channel(1)`.

Signals support arithmetic operations (add, sub, mul, truediv, neg ['-sig' inverts phase]):
```python
>>> sig2 = sig * 2
>>> sig2[-1,1]
2.0
```

**Generating sounds**
All sound generating methods can be used with duration arguments in samples (int) or seconds (float).
One can also set the number of channels by setting the keyword argument nchannels to the desired value.
See doc string in respective function.
Some examples:

- tone(frequency, duration, phase=0, samplerate=None, nchannels=1)
- harmoniccomplex(f0, duration, amplitude=1, phase=0, samplerate=None, nchannels=1)
- whitenoise(duration, samplerate=None, nchannels=1)
- powerlawnoise(duration, alpha, samplerate=None, nchannels=1,normalise=False)
- click(duration, peak=None, samplerate=None, nchannels=1)
- clicktrain(duration, freq, peak=None, samplerate=None, nchannels=1)
- silence(duration, samplerate=None, nchannels=1)
- vowel(vowel='a', gender=''male'', duration=1., samplerate=None, nchannels=1)

**Reading, writing and playing**
```python
>>> sig = slab.Sound.tone(500, 8000, samplerate=8000)
>>> sig.write('tone.wav')
>>> sig2 = slab.Sound('tone.wav')
>>> print(sig2)
<class 'slab.sound.Sound'> duration 1.0, samples 8000, channels 1, samplerate 8000
>>> sig.play()
```

**Timing and sequencing**

- sequence(*sounds, samplerate=None)
- sig.repeat(n)
- sig.ramp(when='both', duration=0.01, envelope=None, inplace=True)

**Plotting**
Examples:
```python
>>> vowel = slab.Sound.vowel(vowel='a', duration=.5, samplerate=8000)
>>> vowel.ramp()
>>> vowel.spectrogram(dyn_range = 50)
>>> Z, freqs, phase = vowel.spectrum(low=100, high=4000, log_power=True)
>>> vowel.waveform(start=0, end=.1)
```

**Binaural sounds**
Binaural is a class for working with binaural sounds, including ITD and ILD manipulation. Binaural inherits all signal generation functions from the Sound class, but returns binaural signals.
Properties:
Binaural.left: left (0th) channel
Binaural.right: right (1st) channel
```python
>>> sig = slab.Binaural.pinknoise(duration=0.5, samplerate=44100)
>>> sig.filter(kind='bp',f=[100,6000])
>>> sig.ramp(when='both',duration=0.15)
>>> sig_itd = sig.itd_ramp(500e-6,-500e-6)
>>> sig_itd.play()
```

**Head-related transfer functions**
This is a class for reading and manipulating head-related transfer functions, essentially a collection of two Filter objects (hrtf.left and hrtf.right) with functions to manage them.
```python
>>> hrtf = HRTF(data='mit_kemar_normal_pinna.sofa') # initialize from sofa file
>>> print(hrtf)
<class 'hrtf.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0
>>> sourceidx = hrtf.cone_sources(20)
>>> hrtf.plot_sources(sourceidx)
>>> hrtf.plot_tf(sourceidx,ear='left')
```

**Psychoacoustic tests:**
The Psychoacoustics classes implement psychophysical procedures and measures, like trial sequences, staircases, and psychometric functions.
```python
>>> tr = Trialsequence(conditions=5, n_reps=2, name='test')
>>> stairs = Staircase(start_val=50, n_reversals=10, step_type='lin', step_sizes=
			[8, 4, 4, 2, 2, 1],  # reduce step size every two reversals
			min_val=0, max_val=60, n_up=1, n_down=1, n_trials=15)
>>> for trial in stairs:
		response = stairs.simulate_response(30)
		print(f'trial # {stairs.this_trial_n}: intensity {trial}, response {response}')
		stairs.add_response(response)
>>> print(f'reversals: {stairs.reversal_intensities}')
>>> print(f'mean of final 6 reversals: {stairs.threshold()}')
>>> stairs.save_json('stairs.json')
>>> stairs.plot()
```
