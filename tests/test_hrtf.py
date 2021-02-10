import slab
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()


def test_hrtf():
    fig, ax = plt.subplots(3)
    fbank = slab.Filter.cos_filterbank()
    hrtf = slab.HRTF(fbank, sources=numpy.random.random(fbank.nfilters))
    hrtf = slab.HRTF("slab/data/mit_kemar_normal_pinna.sofa")
    hrtf.diffuse_field_avg()
    hrtf.diffuse_field_equalization()
    hrtf.elevation_sources()
    sources = hrtf.cone_sources(cone=30)
    hrtf.plot_tf(sources, ear="both", kind="waterfall", show=False, axis=ax[0])
    hrtf.plot_tf(sources, ear="both", kind="image", show=False, axis=[ax[1], ax[2]])
    hrtf.tfs_from_sources(sources)
    hrtf.vsi(sources)
    ax3d = Axes3D(plt.figure())
    hrtf.plot_sources(axis=ax3d, show=False)
