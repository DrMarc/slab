
**slab**: easy manipulation of sounds and psychoacoustic experiments in Python
==============================================================================


**Slab** ('es-lab', or sound laboratory) is an open source project and Python package that makes working with sounds and running psychoacoustic experiments simple, efficient, and fun! For instance, it takes just eight lines of code to run a pure tone audiogram using an adaptive staircase: :ref:`audiogram`



Why slab?
---------
The package aims to lower the entrance barrier for working with sounds in Python and provide easy access to typical operations in psychoacoustics, specifically for students and researchers in the life sciences. The typical BSc or MSc student entering our lab has limited programming and signal processing training and is unable to implement a psychoacoustic experiment from scratch within the time limit of a BSc or MSc thesis. Slab solves this issue by providing easy-to-use building blocks for such experiments. The implementation is well documented and sufficiently simple for curious students to understand. All functions provide sensible defaults and will in many cases 'just work' without arguments (`vowel = slab.Sound.vowel()` gives you a 1-second synthetic vowel 'a', `vowel.spectrogram()` plots the spectrogram). This turned out to be useful for teaching and demonstrations. Many students in our lab have now used the package to implement their final projects and exit the lab as proficient Python programmers.

.. _installation:

Installation
------------

Install the current stable release from the python package index with pip::

    pip install slab

or get the latest development version directly from GitHub (if you have `git <https://git-scm.com>`_) by running::

    pip git+https://github.com/DrMarc/slab.git

**The current version of slab is** |version|.

The releases use `semantic versioning <https://semver.org>`_: ``major.minor.patch``, where ``major`` increments for changes that break backwards compatibility, ``minor`` increments of added functionality, and ``patch`` increases for internal bug fixes.
```slab.__version__``` prints the installed version.

To run the tests::

    pip install slab[testing]

Then go to the installation directory and run::

    pytest

On Linux, you may need to install libsndfile (required by SoundFile) using your distribution's package manager, for instance::

    sudo apt-get install libsndfile1

On Windows, you may need to install `windows-curses <https://pypi.org/project/windows-curses/>`_ (required for getting button presses in the psychoacoustics classes)::

    pip install windows-curses

Working with head related transfer functions requires the h5netcdf module (trying to load a hrtf file will raise an error and tell you to install::

    pip install h5netcdf

All other dependencies should have been automatically installed, and you should see meaningful errors if that did not happen for some reason. The dependencies are: numpy, scipy.signal (for filtering and several other DSP functions), matplotlib (for all plotting), SoundFile (for reading and writing wav files), curses or windows-curses (for getting key presses), and SoundCard (for playing and recording sounds). We have seen a hard-to-replicate problem on some Macs with the SoundCard module: a pause of several seconds after a sound is played. If you experience this issue, just uninstall SoundCard::

    pip uninstall SoundCard

Slab will then use another method to play sounds (winsound on Windows, afplay on Macs, and `SoX <http://sox.sourceforge.net>`_ on Linux), and will record sounds from the microphone using SoX. There are many other packages to play sounds, depending on our operating system. If you prefer a different one, you can easily modify or replace the :meth:`~slab.Sound.play` method.

.. toctree::
  :caption: Contents
  :maxdepth: 2

  introduction
  sounds
  psychoacoustics
  filter
  hrtf
  Worked examples <examples>
  Reference documentation <reference>

**Index of functions and classes:** :ref:`genindex`

**Search the documentation:** :ref:`search`
