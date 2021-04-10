import slab
import numpy
from matplotlib import pyplot as plt
plt.ioff()


def test_create_hrtf():
    hrtf1 = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    assert hrtf1.data[0].fir is True
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


def test_plot_hrtf():
    hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    for ear in ["left", "right", "both"]:
        if ear == "both":
            _, ax = plt.subplots(2)
        else:
            _, ax = plt.subplots(1)
        for kind in ["waterfall", "image"]:
            sources = hrtf.cone_sources(cone=numpy.random.uniform(-180, 180))
            hrtf.plot_tf(sourceidx=sources, kind=kind, ear=ear, axis=ax, show=False)


def test_diffuse_field():
    hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    dfs = hrtf.diffuse_field_avg()
    assert dfs.fir is False
    assert dfs.n_frequencies == hrtf.data[0].n_taps
    equalized = hrtf.diffuse_field_equalization()
    for i in range(hrtf.n_sources):
        _, mag_raw = hrtf.data[i].tf(show=False)
        _, mag_eq = equalized.data[i].tf(show=False)
        assert numpy.abs(mag_eq.mean()) < numpy.abs(mag_raw.mean())


def test_cone_sources():  # this is not working properly!
    hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    sound = slab.Binaural.whitenoise(samplerate=hrtf.samplerate)
    for _ in range(10):
        cone = numpy.random.uniform(-1000, 1000)
        idx = hrtf.cone_sources(cone)
        filtered = [hrtf.data[i].apply(sound) for i in idx]
        ilds = [f.ild() for f in filtered]
        assert numpy.abs(numpy.diff(ilds)).max() < 20


def test_elevation_sources():
    hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    elevations = numpy.unique(hrtf.sources[:, 1])
    elevations = numpy.concatenate([elevations, [-35, 3, 21]])
    for e in elevations:
        sources = hrtf.elevation_sources(e)
        if e in numpy.unique(hrtf.sources[:, 1]):
            assert all(numpy.logical_or(hrtf.sources[sources][:, 0] <= 90, hrtf.sources[sources][:, 0] >= 270))
            assert all(hrtf.sources[sources][:, 1] == e)
        else:
            assert len(sources) == 0


def test_tf_from_sources():
    hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    for i in range(10):
        n_sources = numpy.random.randint(1, 100)
        n_bins = numpy.random.randint(100, 500)
        sources = numpy.random.choice(range(len(hrtf.sources)), n_sources)
        tfs = hrtf.tfs_from_sources(sources, n_bins=n_bins)
        assert tfs.shape[0] == n_bins
        assert tfs.shape[1] == n_sources


def test_vsi():
    hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
    vsi = hrtf.vsi()
    numpy.testing.assert_almost_equal(vsi, 0.73, decimal=2)
    vsis = []
    for i in range(10):
        sources = hrtf.cone_sources(cone=numpy.random.uniform(-180, 180))
        vsis.append(hrtf.vsi(sources=sources))
    assert all(numpy.logical_and(0.4 < numpy.array(vsis), numpy.array(vsis) < 1.1))


def test_plot_sources():
    for i in range(10):
        hrtf = slab.HRTF(slab.DATAPATH+"mit_kemar_normal_pinna.sofa")
        idx = numpy.random.choice(range(len(hrtf.sources)), numpy.random.randint(10))
        hrtf.plot_sources(idx=idx, show=False)
