---
title: 's(ound)lab: A teaching-oriented Python package for designing and running psychoacoustic experiments.'
tags:
  - Python
  - psychoacoustics
  - audio
  - signal processing
  - teaching
authors:
  - name: Marc Schönwiesner
    orcid: 0000-0002-2023-1207
    affiliation: "1, 2"
  - name: Ole Bialas
    orcid:
    affiliation: 1
affiliations:
- index: 1
  name: Institute of Biology, Department of Lifesciences, Leipzig University, Germany
- index: 2
  name: Institute of Psychology, Faculty of Arts and Sciences, University of Montreal, Canada

date: 30 September 2020
bibliography: paper.bib

---
# Summary
Most life science undergraduate students join our lab without prior training in computer programming and digital signal processing.
The primary aim of slab is to enable these students to learn Python, implement novel psychoacoustic experiments, and ultimately complete their theses on time.
To enable quick implementation of experiments, slab implements many of the procedures for psychoacoustic research and experiment control and is easily combined with other Python software. To encourage Python learning, slab provides building blocks rather than ready-made solutions, so that students still need to carefully consider stimulation, sequencing and data management. This also makes slab very flexible and easy to customize. In the documentation (see soundlab.readthedocs.io), we provide tutorials suitable for new students. We also provide actual experiments conducted in our lab using slab as worked examples.

Slab can:
* generate and manipulate single- and multi-channel sounds
* analyse sound by extracting basic sound features
* aid experimental design through stimulus sequence management and response simulation
* calibrate the experimental setup (loudness calibration and frequency equalization)
* display and manipulate head-related transfer functions

Below is an example script that estimates the detection threshold for a 500-Hz pure tone using a staircase procedure. It illustrates the use of the `Staircase` class to manage the staircase. The method `present_tone_trial` is a higher-level convenience function to present a sound and acquire a response from the participant, but each of these steps can be performed separately in a line of code or two when implementing non-standard paradigms.
```
# replace with moving ITD stim!
stimulus = slab.Sound.tone(frequency=500, duration=0.5)
stairs = slab.Staircase(start_val=50, n_reversals=18)
for level in stairs:
    stimulus.level = level
    stairs.present_tone_trial(stimulus)
print(stairs.threshold(n=14))
```

# -> update docs for worked examples
# Statement of need
Slab was written to address our own need for a Python package that allows incoming students to implement their own experiments with clean and maintainable code. Students and researchers should be able to write and understand the code that they are using. Several students have now learned Python and completed their theses using slab, and we think the package may be useful to others in the same situation. Our approach differs from existing software packages for running behavioral experiments, which provide a high level graphical user interface to customize the parameters of pre-made experiments (@psychopy2; @pychoacoustics). In our experience, this leads to very little generalizable learning of Python and experimental control. Slab facilitates this learning by providing basic building blocks, implemented concisely in pure Python , that can be used to construct experiments of various levels of complexity.
There is some overlap with librosa (@librosa), a Python package for music analysis, but that package focusses on feature extraction and does not support psychoacoustic experimentation.
Slab is also one of very few that features manipulation of head-related transfer functions and a simple API for reading a standard file format (SOFA) for such data. There is overlap with more recent implementations of the complete SOFA API (@pysofaconventions, @python-sofa), but these packages provide no methods for typical experimental manipulations of head-related transfer functions. We will likely use `pysofaconventions` internally for handling SOFA files within `slab` in the near future.
The architecture of the `Signal` class and some of the sound generation methods in the `Sound` class are inspired on the 1.4 version of Brian.hears (@brian2hears), but we made several simplifications based on learning reports from students. For instance, signal objects do not implement buffering and do not inherit from Numpy arrays (@numpy) directly, because these features significantly hindered students' understanding of the code.

# Audience
Slab is directed towards students and researchers of all levels studying the perception of sound.
Researchers and incoming students at our lab use it routinely in behavioral and neuroimaging experiments, and the package and has been used in several graduate courses psychophysics and auditory neuroscience.

# References
see paper.bib
