
**slab**: easy manipulation of sounds and psychoacoustic experiments in Python
==============================================================================


**Slab** ('es-lab', or sound laboratory) is an open source project and Python package that makes working with sounds and running psychoacoustic experiments simple, efficient, and fun! For instance, it takes just eight lines of code to run a pure tone audiogram using an adaptive staircase: :ref:`calibration`



Why slab?
---------
The package aims to lower the entrance barrier for working with sounds in Python and provide easy access to typical operations in psychoacoustics, specifically for students and researchers in the life sciences. The typical BSc or MSc student entering our lab has limited programming and signal processing training and is unable to implement a psychoacoustic experiment from scratch within the time limit of a BSc or MSc thesis. Slab solves this issue by providing easy-to-use building blocks for such experiments. The implementation is well documented and sufficiently simple for curious students to understand. All functions provide sensible defaults and will in many cases 'just work' without arguments (`vowel = slab.Sound.vowel()` gives you a 1-second synthetic vowel 'a', `vowel.spectrogram()` plots the spectrogram). This turned out to be useful for teaching and demonstrations. Many students in our lab have now used the package to implement their final projects and exit the lab as proficient Python programmers.

.. _installation:

Installation
------------

Install the current stable release from the python package index with pip:
``pip install soundlab``

or get the latest development version directly from GitHub (if you have `git <https://git-scm.com>`_) by running:
``pip git+https://github.com/DrMarc/soundlab.git``

**The current version is** |version|.

The releases use `semantic versioning <https://semver.org>`_: ``major.minor.patch``, where ``major`` increments for changes that break backwards compatibility, ``minor`` increments of added functionality, and ``patch`` increases for internal bug fixes.
```slab.__version__``` prints the installed version.

Some functionality requires additional dependencies, including playing sounds, saving wav files, reading HRTF files, getting button presses from the keyboard. These are not installed by the package manager, because not everyone may need these functions. The respective methods will raise errors and tell you which dependencies you need and how to install them from the command line. If you prefer, you can install them right away::

    pip install SoundFile SoundCard h5netcdf

On Linux, you need to install libsndfile (required by SoundFile) using your distribution's package manager, for instance::

    sudo apt-get install libsndfile1

Getting single button presses from the keyboard requires the curses library, which unfortunately has different names for Unix-based and Windows operating systems. On a Mac and on Linux you can install::

    pip install curses # Mac and Linux
    pip install windows-curses # Windows

Numpy, scipy.signal (required for filtering and several other DSP functions), and Matplotlib (required for all plotting) should have been automatically installed with slab, and you should see meaningful errors if that did not happen for some reason.

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
