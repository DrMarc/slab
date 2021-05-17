---
name: Bug report
about: Report a bug and help us improve slab
title: "[BUG in class ...]"
labels: ''
assignees: DrMarc

---

**Describe the bug**
A clear and concise description of what the bug is.
For instance: "Binaural.pinknoise() should return a stereo sound (two channels), but instead returns a mono sound (one channel)."

**To Reproduce**
Minimal code example. This should ideally run as is, without additional data. Include all imports.
For instance:
> import slab
> sig = slab.Binaural.pinknoise()
> sig.n_channels

**Expected behaviour**
The expected output of the minimal code example. For instance, in the example above:
> 2

**Actual behaviour**
The output of the minimal code example. For instance, in the example above:
> 1

**OS and Version:**
 - OS: [e.g. MacOS, Win, Linux]
 - Version [e.g. v0.9]
