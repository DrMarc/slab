.. _HRTF:

HRTFs
=====

The :class:`HRTF` class provides methods for manipulating, plotting, and applying head-related transfer functions.

.. note:: The class is at the moment geared towards plotting and analysis of HRTF files in the `sofa format <https://www.sofaconventions.org/>`_, because we needed that functionality for grant applications. The functionality will grow as we start to record and manipulate HRTFs more often.

.. note:: When we started to writing this code, there was no python module for reading and writing sofa files. Now that `pysofaconventions <https://github.com/andresperezlopez/pysofaconventions>`_ is available, we will at some point switch internally to using that module as backend for reading sofa files, instead of our own limited implementation.

The standard HRTF recordings from the KEMAR acoustic mannequin are included in the :data:`DATAPATH` folder of the module. Libraries of many other recordings can be found on the `website of the sofa file format <https://www.sofaconventions.org/>`_. Read the HRTF file by calling the :class:`HRTF` class with the file name as :attr:`data` argument. Print the resulting object to obtain information about the structure of the HRTF data:

.. plot::
    :include-source:
    :nofigs:
    :context:

    from slab import DATAPATH
    hrtf = slab.HRTF(data=DATAPATH+'mit_kemar_normal_pinna.sofa')
    print(hrtf)
    # <class 'hrtf.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0

Plot the source positions in 3D with the :meth:`.plot_sources` to get an impression of the density of the recordings. If you supply a list of source indices, then these sources are highlighted in red. Two methods allow you to select sources on vertical and horizontal arcs: the :meth:`.elevation_sources` selects sources along a horizontal sphere slice at a given elevation, and the :meth:`.cone_sources` selects sources along a vertical slice through the source sphere in front of the listener a given angular distance away from the midline (the arc at zero in the example below selects sources on the vertical midline - try a few angles to understand how the argument works):

.. plot::
    :include-source:
    :context:

    sourceidx = hrtf.cone_sources(0) # select sources on a cone of confusion at 5 deg from midline
    hrtf.plot_sources(sourceidx) # plot the sources in 3D, highlighting the selected sources

Now that you have selected some sources, you can inspect their transfer functions with the :meth:`.plot_tf`, which uses the :meth:`~slab.Filter.tf` method of the underlying :class:`Filter` objects. Before plotting, we apply a diffuse field equalization to remove non-spatial components of the HRTF, which makes the features of the HRTF that change with direction easier to see:

.. plot::
    :include-source:
    :context: close-figs

    hrtf.diffuse_field_equalization()
    hrtf.plot_tf(sourceidx, ear='left')

The image above is a `waterfall` plot as in Wightman and Kistler, 1989, and below is an `image` plot as in Hofman 1998:

.. plot::
    :include-source:
    :context: close-figs

    hrtf.plot_tf(sourceidx, ear='left', kind='image')
