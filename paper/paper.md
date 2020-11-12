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
Typically, undergrads joining our lab for a research project are faced with a lot of issues that their
academic studies did not prepare them for - like coding and digital signal processing. The looming deadline
and a steep learning curve can often leave the student feeling overwhelmed. We designed slab to provide a remedy for 
this situation. Our aim is to make it easier for students to enter research while also facilitating their learning 
process. Slab contains all of the fundamental tools necessary for studying the perception of sound and is easily 
combined with other Python software.Rather than implement a high-level interface for all of those features, we equip the
user with basic building blocks. This requires the user to make careful considerations about stimulation, sequencing
and data management. It also makes slab very flexible and easy to customize. In the documentation 
(see soundlab.readthedocs.io), we provide tutorials on all functionalities. In addition to that, we regularly upload
experiments that we conducted using slab to the repository (github.com/DrMarc/soundlab).

The functionalities include:
* Generating and manipulating sounds
* Experimental Design
* Frequency filtering and equalization
* Binaural sounds
* Handling head-related transfer functions
* Basic sound-feature extraction

# Statement of need
We believe that researchers should be able to write and understand the code that they are using. We want to make this
possible, despite the lack of formal training and time constraints which undergrads are typically facing.
We want to facilitate learning by providing basic building blocks and instructing researchers on how to combine them 
to experiments of various levels of complexity. Our approach differs from other software packages for running
behavioral experiments which provide a high level graphical user interface to customize the parameters of experiments
(@psychopy2_2019; @pychoacoustics). While there is some overlap with sound and music processing packages like librosa
(@librosa) we only implemented the basics of sound processing and analysis. While there is a Python API for the
spatially oriented format for acoustics, there is, to our knowledge, no package that features experimental
manipulation of head-related transfer functions. The signal class is based on BrianHears (@brian2hears).


# Audience
Slab is directed towards students and researchers of all levels studying the perception of sound. 
It is routinely used at the Neurobiology department at the University of Leipzig in behavioral experiments as well as
in combination with electrophysiology and neuroimaging.

# References
see paper.bib
