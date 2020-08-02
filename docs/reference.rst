.. currentmodule:: slab

Reference documentation
=======================

.. warning:: This reference documentation is auto-generated from the doc strings in the module. For a tutorial-like overview of the functionality of slab, please see the previous sections.

Sounds
^^^^^^
Inherits from :class:`slab.Signal`.

.. autoclass:: Sound
   :members:

Signal
------
:class:`slab.Sound` inherits from Signal, which provides basic methods to handle signals:

.. autoclass:: Signal
   :members:

Binaural sounds
---------------
Binaural sounds inherit from Sound and provide methods for manipulating interaural parameters of two-channel sounds.

.. autoclass:: sounds.Binaural
  :members:

Psychoacoustic procedures
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Trialsequence
   :members:

.. autoclass:: Staircase
   :members:

.. autoclass:: Precomputed
   :members:

Filters
-------
.. autoclass:: Filter
   :members:

HRTFs
-----
.. autoclass:: HRTF
   :members:
