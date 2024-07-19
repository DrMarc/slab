.. currentmodule:: slab

.. _Reference:

Reference documentation
=======================

.. note:: This reference documentation is auto-generated from the doc strings in the module. For a tutorial-like overview of the functionality of slab, please see the previous sections.

Sounds
^^^^^^
Inherits from :class:`slab.Signal`.

.. autoclass:: Sound
   :members:
   :member-order: bysource

.. automethod:: slab.sound.apply_to_path

Signal
------
:class:`slab.Sound` inherits from Signal, which provides basic methods to handle signals:

.. autoclass:: Signal
   :members:
   :member-order: bysource

Binaural sounds
---------------
Binaural sounds inherit from Sound and provide methods for manipulating interaural parameters of two-channel sounds.

.. autoclass:: Binaural
  :members:
  :member-order: bysource

Psychoacoustic procedures
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Trialsequence
   :members:
   :inherited-members:
   :member-order: bysource

.. autoclass:: Staircase
   :members:
   :inherited-members:
   :member-order: bysource

.. autoclass:: Precomputed
   :members:
   :member-order: bysource

.. autoclass:: ResultsFile
   :members:
   :member-order: bysource

.. automethod:: slab.psychoacoustics.key

.. automethod:: slab.psychoacoustics.load_config


Filters
^^^^^^^
.. autoclass:: Filter
   :members:
   :member-order: bysource

HRTFs
^^^^^
.. autoclass:: HRTF
   :members:
   :member-order: bysource

.. autoclass:: Room
   :members:
   :member-order: bysource
