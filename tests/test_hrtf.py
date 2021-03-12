import slab
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ioff()


def test_create_hrtf():
    hrtf1 = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    hrtf2 = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    for i in range(len(hrtf1.data)):
        numpy.testing.assert_equal(hrtf1.data[i].data, hrtf2.data[i].data)
    del hrtf2
    for i in range(100):
        idx = numpy.random.choice(range(hrtf1.n_sources))
        data = hrtf1.data[idx]  # make HRTF from instance of filter
        numpy.testing.assert_raises(ValueError, slab.HRTF, data)
        source = hrtf1.sources[idx]
        listener = numpy.random.randn(3)
        hrtf = slab.HRTF(data=data, sources=source, listener=listener)
        numpy.testing.assert_equal(hrtf.listener, listener)
        numpy.testing.assert_equal(hrtf.sources, source)
        numpy.testing.assert_equal(hrtf.data[0].data.flatten(), data.data[:, 0])
        numpy.testing.assert_equal(hrtf.data[1].data.flatten(), data.data[:, 1])
        idx = numpy.random.choice(range(hrtf1.n_sources), 10, replace=False)
        data = [hrtf1.data[i].data for i in idx]  # make HRTF from array
        data = numpy.dstack(data)
        data = numpy.transpose(data, axes=(2, 0, 1))
        sources = hrtf1.sources[idx]
        hrtf = slab.HRTF(data=data, sources=sources)
        assert hrtf.n_sources == data.shape[0]
        assert hrtf.data[0].n_samples == data.shape[1]
        assert hrtf.data[0].n_filters == data.shape[2]

