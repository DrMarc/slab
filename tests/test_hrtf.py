import slab
import numpy


def test_hrtf():

    fbank = slab.Filter.cos_filterbank()
    hrtf = slab.HRTF(fbank, sources=numpy.random.random(fbank.nfilters))
    hrtf = slab.HRTF("slab/data/mit_kemar_normal_pinna.sofa")
    hrtf.diffuse_field_avg()
    hrtf.diffuse_field_equalization()
    hrtf.elevation_sources()
    sources = hrtf.cone_sources(cone=30)
    hrtf.plot_tf(sources, ear="both", kind="waterfall")
    hrtf.plot_tf(sources, ear="both", kind="image")
    hrtf.tfs_from_sources(sources)
    hrtf.vsi(sources)
    hrtf.plot_sources()
