# slab

Class for working with sounds, including loading/saving, manipulating and playing

* Signals
* Sound (inherits from Signals, methods for generating, manipulating, displaying, and analysing sound stimuli)
* Binaural (inherits from Sound, contains convenience functions for binaural stimuli)
* Filter (inherits from Signals, a stub at the moment)
* HRTF (inherits from Filter, quick interface to the .sofa format for spatial audio and functions for working with binaural filter functions)
* Psychoacoustics (a collection of classes for working trial sequences and psychoacoustic testing)


Much of the basic functionality and software architecture of the Signal and Sound classes is based on [brian.hears] (www.briansimulator.org/docs/hears.html). The classes are used primarily to lower the entrance barrier for working with sounds in Python for students in our lab, and provide easy access to typical operations in psychoacoustics.

```python
>>> import slab
>>> sig = slab.Sound.tone()
>>> sig.level = 80
```
