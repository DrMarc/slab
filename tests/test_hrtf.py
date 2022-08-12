import slab
import numpy
from matplotlib import pyplot as plt
plt.ioff()


def test_create_hrtf():
    hrtf1 = slab.HRTF.kemar()
    assert hrtf1.data[0].fir is True
    hrtf2 = slab.HRTF.kemar()
    for i in range(len(hrtf1.data)):
        numpy.testing.assert_equal(hrtf1[i].data, hrtf2[i].data)
    del hrtf2
    for i in range(100):
        idx = numpy.random.choice(range(hrtf1.n_sources))
        data = hrtf1[idx]  # make HRTF from instance of filter
        numpy.testing.assert_raises(ValueError, slab.HRTF, data)
        source = hrtf1.sources.vertical_polar[idx]
        listener = numpy.random.randn(3)
        hrtf = slab.HRTF(data=data, sources=source, listener=listener)
        numpy.testing.assert_equal(hrtf.listener, listener)
        numpy.testing.assert_equal(hrtf.sources.vertical_polar, source[numpy.newaxis, ...].astype('float16'))
        numpy.testing.assert_equal(hrtf[0].data.flatten(), data.data[:, 0])
        numpy.testing.assert_equal(hrtf[1].data.flatten(), data.data[:, 1])
        idx = numpy.random.choice(range(hrtf1.n_sources), 10, replace=False)
        data = [hrtf1[i].data for i in idx]  # make HRTF from array
        data = numpy.dstack(data)
        data = numpy.transpose(data, axes=(2, 0, 1))
        sources = hrtf1.sources.cartesian[idx]
        hrtf = slab.HRTF(data=data, sources=sources, samplerate=hrtf.samplerate, datatype='FIR')
        assert hrtf.n_sources == data.shape[0]
        assert hrtf[0].n_samples == data.shape[1]
        assert hrtf[0].n_filters == data.shape[2]


def test_plot_hrtf():
    hrtf = slab.HRTF.kemar()
    for ear in ["left", "right", "both"]:
        if ear == "both":
            _, ax = plt.subplots(2)
        else:
            _, ax = plt.subplots(1)
        for kind in ["waterfall", "image"]:
            cone = numpy.random.uniform(-180, 180)
            sources = hrtf.cone_sources(cone=cone)
            # sources = hrtf.cone_sources(cone=numpy.random.uniform(-180, 180))
            hrtf.plot_tf(sourceidx=sources, kind=kind, ear=ear, axis=ax, show=False)


def test_diffuse_field():
    hrtf = slab.HRTF.kemar()
    dfs = hrtf.diffuse_field_avg()
    assert dfs.fir is False
    assert dfs.n_frequencies == hrtf.data[0].n_taps
    equalized = hrtf.diffuse_field_equalization()
    for i in range(hrtf.n_sources):
        _, mag_raw = hrtf.data[i].tf(show=False)
        _, mag_eq = equalized.data[i].tf(show=False)
        assert numpy.abs(mag_eq.mean()) < numpy.abs(mag_raw.mean())


def test_cone_sources():  # this is not working properly!
    hrtf = slab.HRTF.kemar()
    sound = slab.Binaural.whitenoise(samplerate=hrtf.samplerate)
    for _ in range(10):
        cone = numpy.random.uniform(-1000, 1000)
        idx = hrtf.cone_sources(cone)
        filtered = [hrtf.data[i].apply(sound) for i in idx]
        ilds = [f.ild() for f in filtered]
        assert numpy.abs(numpy.diff(ilds)).max() < 20


def test_elevation_sources():
    hrtf = slab.HRTF.kemar()
    elevations = numpy.unique(hrtf.sources.vertical_polar[:, 1])
    elevations = numpy.concatenate([elevations, [-35, 3, 21]])
    for e in elevations:
        sources = hrtf.elevation_sources(e)
        if e in numpy.unique(hrtf.sources.vertical_polar[:, 1]):
            assert all(numpy.logical_or(hrtf.sources.vertical_polar[sources][:, 0] <= 90,
                                        hrtf.sources.vertical_polar[sources][:, 0] >= 270))
            assert all(hrtf.sources.vertical_polar[sources][:, 1] == e)
        else:
            assert len(sources) == 0


def test_tf_from_sources():
    hrtf = slab.HRTF.kemar()
    for _ in range(10):
        n_sources = numpy.random.randint(1, 100)
        n_bins = numpy.random.randint(100, 500)
        sources = numpy.random.choice(range(len(hrtf.sources.vertical_polar)), n_sources)
        tfs = hrtf.tfs_from_sources(sources, n_bins=n_bins)
        assert tfs.shape[0] == n_bins
        assert tfs.shape[1] == n_sources


def test_vsi():
    hrtf = slab.HRTF.kemar()
    vsi = hrtf.vsi()
    numpy.testing.assert_almost_equal(vsi, 0.73, decimal=2)
    vsis = []
    for _ in range(10):
        sources = hrtf.cone_sources(cone=numpy.random.uniform(-180, 180))
        vsis.append(hrtf.vsi(sources=sources))
    assert all(numpy.logical_and(numpy.array(vsis) > 0.4, numpy.array(vsis) < 1.1))


def test_plot_sources():
    hrtf = slab.HRTF.kemar()
    for _ in range(10):
        idx = numpy.random.choice(range(len(hrtf.sources.vertical_polar)), numpy.random.randint(10))
        hrtf.plot_sources(idx=idx, show=False)


def test_interpolate():
    hrtf = slab.HRTF.kemar()
    for _ in range(10):
        idx = numpy.random.choice(range(len(hrtf.sources.vertical_polar)))
        azi, ele = hrtf.sources.vertical_polar[idx][0:2]
        method = numpy.random.choice(['nearest', 'bary'])
        h = hrtf.interpolate(azimuth=azi, elevation=ele, method=method)
        _, spec_interp = h[0].tf(show=False)
        _, spec_origin = hrtf[idx].tf(show=False)
        nearer_channel = 0 if azi-180 < 0 else 1
        assert numpy.corrcoef(spec_interp[:, nearer_channel], spec_origin[:, nearer_channel]).min() > 0.99


def test_convert_coordinates():
    hrtf = slab.HRTF.kemar()
    vertical_polar = hrtf.sources.vertical_polar
    cartesian_from_vertical = slab.HRTF._vertical_polar_to_cartesian(vertical_polar)
    interaural_from_vertical = slab.HRTF._vertical_polar_to_interaural_polar(vertical_polar)
    cartesian_from_interaural = slab.HRTF._interaural_polar_to_cartesian(interaural_from_vertical)
    numpy.testing.assert_almost_equal(cartesian_from_interaural, cartesian_from_vertical, decimal=3)
    vertical_from_cartesian = slab.HRTF._cartesian_to_vertical_polar(cartesian_from_vertical)
    numpy.testing.assert_almost_equal(vertical_from_cartesian, vertical_polar, decimal=3)


def test_estimate_hrtf():
    fs = 44800
    duration = 1.0
    signal1 = slab.Sound.whitenoise(duration=duration, samplerate=fs)
    n_sources = numpy.random.randint(360)
    azimuths = numpy.random.choice(range(0, 360), size=n_sources, replace=False)
    elevations = numpy.random.choice(range(-180, 180), size=n_sources, replace=False)
    recordings = []
    sources = []
    for i in range(len(azimuths)):
        sources.append([azimuths[i], elevations[i], 1])
        recordings.append(slab.Binaural.whitenoise(duration = duration, samplerate=fs))
    hrtf1 = slab.HRTF.estimate_hrtf(recordings, signal1, sources)
    assert hrtf1[0].samplerate == recordings[0].samplerate
    assert hrtf1[0].n_samples == fs / 2 + 1
    assert hrtf1[0].n_filters == 2
    assert hrtf1.n_sources == n_sources
    assert hrtf1.n_elevations == n_sources
    signal2 = slab.Sound.whitenoise(duration=duration-0.1, samplerate=fs-800)
    hrtf2 = slab.HRTF.estimate_hrtf(recordings, signal2, sources)
    assert hrtf1.samplerate == hrtf2.samplerate
    assert hrtf1[0].n_samples == hrtf2[0].n_samples
