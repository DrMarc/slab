

HRTFs
=====

**HRTF**: Inherits from Filter, reads .sofa format HRTFs and provides methods for manipulating, plotting, and applying head-related transfer functions.::

    hrtf = slab.HRTF(data='mit_kemar_normal_pinna.sofa') # load HRTF from a sofa file (the standard KEMAR data is included)
    print(hrtf) # print information
    <class 'hrtf.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0
    sourceidx = hrtf.cone_sources(20) # select sources on a cone of confusion at 20 deg from midline
    hrtf.plot_sources(sourceidx) # plot the sources in 3D, highlighting the selected sources
    hrtf.plot_tf(sourceidx,ear='left') # plot transfer functions of selected sources in a waterfall plot
	hrtf.diffuse_field_equalization() # apply diffuse field equalization to remove non-spatial components of the HRTF
