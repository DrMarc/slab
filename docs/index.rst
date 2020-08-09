
**slab**: easy manipulation of sounds and psychoacoustic experiments in Python
==============================================================================

**Slab** ('es-lab', or sound laboratory) is an open source project and Python package that makes working with sounds and running psychoacoustic experiments simple, efficient, and fun! For instance, it takes just eight lines of code to run a pure tone audiogram using an adaptive staircase: ::

    import slab
    stimulus = slab.Sound.tone(frequency=500, duration=0.5) # make a 0.5 sec pure tone of 500 Hz
    stairs = slab.Staircase(start_val=50, n_reversals=10) # set up the adaptive staircase
    for level in stairs: # the staircase object returns a value between 0 and 50 dB for each trial
        stimulus.level = level
        stairs.present_tone_trial(stimulus) # plays the tone and records a keypress (1 for 'heard', 2 for 'not heard')
        stairs.print_trial_info() # optionally print information about the current state of the staircase
    print(stairs.threshold()) # print threshold when done

Why slab?
---------
The package aims to lower the entrance barrier for working with sounds in Python and provide easy access to typical operations in psychoacoustics, specifically for students and researchers in the life sciences. The typical BSc or MSc student entering our lab has limited programming and signal processing training and is unable to implement a psychoacoustic experiment from scratch within the time limit of a BSc or MSc thesis. Slab solves this issue by providing easy-to-use building blocks for such experiments. The implementation is well documented and sufficiently simple for curious students to understand. All functions provide sensible defaults and will in many cases 'just work' without arguments (`vowel = slab.Sound.vowel()` gives you a 1-second synthetic vowel 'a', `vowel.spectrogram()` plots the spectrogram). This turned out to be useful for teaching and demonstrations. Many students in our lab have now used the package to implement their final projects and exit the lab as proficient Python programmers.

Installation
------------

Install the current stable release from the python package index with pip:
``pip install soundlab``

or get the latest development version directly from GitHub (if you have `git <https://git-scm.com>`_) by running:
``pip git+https://github.com/DrMarc/soundlab.git``

**The current version is** |version|.

The releases use `semantic versioning <https://semver.org>`_: ``major.minor.patch``, where ``major`` increments for changes that break backwards compatibility, ``minor`` increments of added functionality, and ``patch`` increases for internal bug fixes.
```slab.__version__``` prints the installed version.

.. toctree::
  :caption: Contents
  :maxdepth: 2
  :titlesonly:

  introduction
  sounds
  psychoacoustics
  filter
  hrtf
  Worked examples <examples>
  Reference documentation <reference>

**Index of functions and classes:** :ref:`genindex`

**Search the documentation:** :ref:`search`
