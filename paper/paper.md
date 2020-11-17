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
To enable quick implementation of experiments, slab implements many of the procedures for psychoacoustic research and experiment control and is easily
combined with other Python software. To encourage Python learning, slab provides building blocks rather than ready-made solutions, so that students still need to carefully consider stimulation, sequencing and data management. This also makes slab very flexible and easy to customize. In the documentation
(see soundlab.readthedocs.io), we provide tutorials suitable for new students. We also provide actual experiments conducted in our lab using slab as worked examples.

Slab can:
* generate and manipulate single- and multi-channel sounds
* analyse sound by extracting basic sound features
* aid experimental design through stimulus sequence management and response simulation
* calibrate the experimental setup (loudness calibration and frequency equalization)
* display and manipulate head-related transfer functions

# Statement of need
Students and researchers should be able to write and understand the code that they are using. We want to make this
possible, despite the lack of formal training and time constraints which undergrads are typically facing.
We want to facilitate learning by providing basic building blocks and instructing researchers on how to combine them
to experiments of various levels of complexity. Our approach differs from other software packages for running
behavioral experiments which provide a high level graphical user interface to customize the parameters of experiments
(@psychopy2_2019; @pychoacoustics). While there is some overlap with sound and music processing packages like librosa
(@librosa) we only implemented the basics of sound processing and analysis. While there is a Python API for the
spatially oriented format for acoustics, there is, to our knowledge, no package that features experimental
manipulation of head-related transfer functions. The signal class is based on Brian.Hears (@brian2hears).


# Audience
Slab is directed towards students and researchers of all levels studying the perception of sound.
Researchers and incoming students at our lab use it routinely in behavioral experiments and neuroimaging experiments.

# References
see paper.bib
