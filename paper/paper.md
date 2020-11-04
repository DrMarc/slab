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
Psychoacoustics is the scientific study of the perception of sound. Designing
and conducting psychoacoustic experiments requires knowledge about managing psychological experiments,
as well as digital signal processing and acoustics. Researchers who lack the math and coding skills
to implement everything from scratch are forced to draw from a number of different packages which is
bound to cause problems. With slab, we provide a framework which is powerful yet accessible
for handling all aspects of an experiment like managing trial sequences, generating and playing sounds,
extracting features, recording responses and saving results. Rather than provide a high-level interface for all of those features,
slab equips the user with the basic building blocks. This modular design makes it easy to integrate your own code and facilitates
learning by forcing the user to make careful considerations about experimental design and signal processing.

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

# Typical slab workflow
Every experiment starts with quesitions about the design: How many conditions exists? How often are they repeated?
What is the order? Do I need to precompute stimuli? Questions like these are handled by the Psychoacoustics module.
This module features precomputed trial sequences, as well as adaptive ones where the trials are computed dynamically
while the experiment is running. It also handles input during the experiment via keyboard or a button-box.


# Typical Psychoacoustics workflow


# Figures

# Acknowledgements

We acknowledge contributions from

# References
