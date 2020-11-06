---
title: 's(ound)lab: A teaching-oriented Python package for running psychoacoustic experiments and manipulating sounds'
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
Designing and conducting psychoacoustic experiments requires skills in coding and digital signal
processing - both of which are usually not taught in a typical psychology undergraduate course.
Researchers who lack the math and coding skills to implement everything from scratch have to draw
from a number of different packages which is bound to cause problems. With slab, we provide a framework
which is powerful yet accessible for handling all aspects of an experiment like managing trial sequences,
generating and playing sounds, extracting features, recording responses and saving results.
Rather than provide a high-level interface for all of those features, slab equips the user with the basic building blocks.
This modular design makes it easy to integrate your own code and facilitates
learning by requiring the user to make careful considerations about stimulation, sequencing and data management.
To make it easier for users to get started we provide tutorials (see soundlab.readthedocs.io) as well as sample
experiments that we conducted in our lab.

The functionalities include:
* Generating and manipulating sounds
* Sound-feature extraction
* Experimental Design
* Frequency filtering and equalization
* Spatial binaural sounds
* Handling head-related transfer functions


# Statement of need
We believe that researchers should be able to write and understand the code they are using, even if they
lack formal training. With slab, we want to facilitate learning by providing basic building blocks and instructing
researchers on how to combine them to experiments of various levels of complexity. This approach sets us apart from
other packages with similar scope (@psychopy2_2019; @pychoacoustics) which are run mainly via a graphic user interface
and come with default experiments that work out of the box. There is considerable overlap with sound and music
processing packages like librosa (@librosa). However, while librosa offers much more in terms of sound analysis and
feature extraction, it has an emphasis on music and lacks the utilities for conducting experiments. The signal class
is based on BrianHears (@brian2hears). To our knowledge, slab is the only python package that features handling of head-realted transfer funtion for experimental purpose. Slab is used in several ongoing experiments partially in combination with
electrophysiology and neuroimaging.


# Audience
Slab is directed towards a broad audience of students and researchers studying the
perception of sound. It is useful for students who are learning how to code and design experiments
as well as for senior researchers who aim at making their code cleaner and more readable.
Slab is routinely used in experiments at the Neurobiology department at the University of Leipzig.


# References
see paper.bib
