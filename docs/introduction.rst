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

* **I have set the level of a sound to 70 dB but it is way louder, why?**

This is because soundlab does not know the system you are using to play sound.
For example: white noise is generated so that the maximum value in the time
series is +1 and the minimum minus one. This signal is assigned a loudness of
82 dB per default but this is not a meaningful value because a +1 to -1 signal
might be interpreted differently by different systems. See :ref:`calibration`
on how to to adjust soundlab to your setup.


* **What is the difference between white noise and pink noise?**

White noise is a signal that consists of random numbers. This signal has equal
power at all frequencies. However, our auditory system does not perceive it that way
which is why white noise appears high-pitched. In the pink noise signal, the power
decreases with frequency to correct for this effect. You can use the function
:func:`Sound.powerlawnoise` to create your own noise.


* **How can i make sure that the sounds I am using are similar in their low-level features?**

There is no conclusive list of sound features that are important but several ones are implemented
in :meth:`spectral_feature`. You could also construct a control condition by extracting the sounds
envelope and generating noise that has the same temporal properties as the original sound.


* **I think I found a bug!**

Please see the [bug reports](https://github.com/user/DrMarc/soundlab/CONTRIBUTING.md#bugs) section in the contribution guidelines.


* **How can I contribute to the project?**

Please see the [pull request](https://github.com/user/DrMarc/soundlab/CONTRIBUTING.md#pull-requests) section in the contribution guidelines if you want to contribute code or useful examples for the documentation.
