Introduction
============

Overview
--------

In this documentation we do not aim at providing a comprehensive explanation of
every single slab function (a complete description can be found in the :ref:`reference` section).
Rather, we want to provide some guidance for you to start generating sounds and running experiments.

For starters, you should have a look at the :ref:`Sounds` section. There, you will learn how to
generate, manipulate and write/read Sounds in slab. Next, you should see the :ref:`Psychoacoustics`
section which is about generating trial sequences and running experiments. With these tools you can
already do plenty of things! For example...

The :ref:`Filter` section contains some more advanced, but powerful, methods for processing
digital signals. The :ref:`HRTF` section describes the handling of head related transfer functions and
will only be relevant if you are interested in spatial audio.


Frequently Asked Questions
--------------------------

* **Where can I learn enough Python to use this module?**

You can find many free courses online. We usually point our students to `Google's Python class <https://developers.google.com/edu/python>`_. For those of you who prefer video, Coursera has two suitable courses: `Python for Everybody <https://www.coursera.org/learn/python>`_ and `An Introduction to Interactive Programming with Python <https://www.coursera.org/learn/interactive-python-1?trk=profile_certification_title>`_.
There are also courses specifically for sound and signal processing, for instance `this one <https://www.coursera.org/learn/audio-signal-processing>`_.


* **Which Python environment do you use in the lab?**

We recommend `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which bundles Python and the conda package manager and installs quickly. You can then install only the packages that you need for your work, like IPython, numpy, scipy, and matplotlib, with a single command::

    conda install ipython numpy scipy matplotlib

When programming we use the command line with IPython and the Atom text editor with a package for syntax highlighting. Some lab members use `PyCharm <https://www.jetbrains.com/pycharm/>`_ or `Spyder <https://www.spyder-ide.org>`_ as integrated development environments. We don't recommend IDEs for beginners, because in our experience, students tend to conflate the IDE with Python itself and develop programming habits that they need to unlearn when they want to get productive.


* **I get import errors when using certain functions!**

Slab requires additional modules for some functionality. These modules are not installed automatically because not everyone may need them (such as HRTF file reading) or the installation is OS-dependent (such as SoundFile and curses). Please see :ref:`installation` for how and what to install should you need it. The import error messages will in most cases give you the necessary installation command for Mac/Linux systems.


* **I have set the level of a sound to 70 dB but it is way louder, why?**

This is because soundlab does not know the hardware you are using to play sound. For example, white noise is generated so that the maximum value in the time series is +1 and the minimum minus one ("full scale"). The RMS of this signal, expressed in deciBels happens to be about 82 dB, but you need to calibrate your system (see :ref:`calibration`) so that the calculated intensity is meaningful. Relative intensities are correct without calibration---so decreasing the intensity by 10 dB (`sound.level -= 10`) will work as expected.


* **What is the difference between white noise and pink noise?**

White noise is a signal that consists of random numbers. This signal has equal power at all frequencies. However, our auditory system does not perceive it that way, which is why white noise appears high-pitched. In the pink noise signal, the power decreases with frequency to correct for this effect. Pink noise is thus a more appropriate choice for a masking or background noise, because it has the same power in each octave. However, there are even better options. The :meth:`~slab.Sound.erb_noise` method constructs a noise with equal energy not in octaves, but in fractions of approximated auditory filters widths (equivalent rectangular bandwidths, ERB). Or the :meth:`~slab.Sound.multitone_masker`, which is a noise-like combination of many pure tones at ERB intervals. This noise does not have random amplitude variations and masks evenly across frequency and time.


* **I think I found a bug!**

Please see the `bug reports <https://github.com/user/DrMarc/soundlab/CONTRIBUTING.md#bugs>`_ section in the contribution guidelines.


* **How can I contribute to the project?**

Please see the `pull request <https://github.com/user/DrMarc/soundlab/CONTRIBUTING.md#pull-requests>`_ section in the contribution guidelines if you want to contribute code or useful examples for the documentation.
