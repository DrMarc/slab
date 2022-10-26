"""
Class for reading and manipulating head-related transfer functions. Reads files in .sofa format (started before
python implementations of the sofa conventions were available -> will be migrated to use pysofaconventions!)
"""

import copy
import warnings
import pathlib
import pickle
import bz2
import numpy
import datetime
try:
    import matplotlib
    from matplotlib import pyplot as plt
except ImportError:
    matplotlib, plt = False, False
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    Axes3D = False
    make_axes_locatable = False
try:
    import h5netcdf
except ImportError:
    h5netcdf = False
try:
    import netCDF4
except ImportError:
    netCDF4 = False
try:
    import scipy.signal
    import scipy.spatial
except ImportError:
    scipy = False
from slab.signal import Signal
from slab.filter import Filter
from slab.sound import Sound
from collections import namedtuple

_kemar = None

class HRTF:
    """
    Class for reading and manipulating head-related transfer functions with attributes and functions to manage them.

    Arguments:
        data (str | Filter | numpy.ndarray): Typically, this is the path to a file in the .sofa format.
            The file is then loaded and the data of each source for which the transfer function was recorded is stored
            as a Filter object in the `data` attribute. Instead of a file name, the data can be passed directly as
            Filter or numpy array. Given a `Filter`, every filter channel in the instance is taken as a source (this
            does not result in a typical HRTF object and is only intended for equalization filter banks). Given a 3D
            array, the first dimension represents the sources, the second the number of taps per filter and the last the
            number of filter channels per filter (should be always 2, for left and right ear).
        datatype (None | string): type of the HRTF filter bank, can be 'FIR' for finite imoulse response filters or 'TF'
            for Fourier filters.
        samplerate (None | float): rate at which the data was acquired, only relevant when not loading from .sofa file
        sources (None | array): positions of the recorded sources, only relevant when not loading from .sofa file
        listener (None | list | dict): position of the listener, only relevant when not loading from .sofa file
        verbose (bool): print out items when loading .sofa files, defaults to False

    Attributes:
        .data (list): The HRTF data. The elements of the list are instances of slab.Filter.
        .datatype (string): Type of the HRTF filter bank.
        .samplerate (int): sampling rate at which the HRTF data was acquired.
        .sources (named tuple): Cartesian coordinates (x, y, z), vertical-polar and interaural-polar coordinates
            (azimuth, elevation, distance) of all sources.
        .n_sources (int): The number of sources in the HRTF.
        .n_elevations (int): The number of elevations in the HRTF.
        .listener (dict): A dictionary containing the position of the listener ("pos"), the point which the listener
            is fixating ("view"), the point 90° above the listener ("up") and vectors from the listener to those points.
    Example:
        import slab
        hrtf = slab.HRTF.kemar() # use inbuilt KEMAR data
        sourceidx = hrtf.cone_sources(20)
        hrtf.plot_sources(sourceidx)
        hrtf.plot_tf(sourceidx, ear='left')
    """
    # instance properties
    n_sources = property(fget=lambda self: len(self.sources.vertical_polar),
                         doc='The number of sources in the HRTF.')
    n_elevations = property(fget=lambda self: len(self.elevations()),
                            doc='The number of elevations in the HRTF.')

    def __init__(self, data, datatype=None, samplerate=None, sources=None, listener=None, verbose=False):
        if isinstance(data, pathlib.Path):
            data = str(data)
        if isinstance(data, str):
            if samplerate is not None:
                raise ValueError('Cannot specify samplerate when initialising HRTF from a file.')
            if pathlib.Path(data).suffix != '.sofa':
                raise NotImplementedError('Only .sofa files can be read.')
            f = HRTF._sofa_load(data, verbose)
            self.data, self.datatype, self.samplerate = HRTF._sofa_get_data(f)
            sources, coordinate_system = HRTF._sofa_get_sources(f)
            self.sources = HRTF._get_coordinates(sources, coordinate_system)
            self.listener = HRTF._sofa_get_listener(f)
        elif isinstance(data, Filter):
            # This is a hacky shortcut for casting a filterbank as HRTF. Avoid unless you know what you are doing.
            if sources is None:
                raise ValueError('Must provide spherical source positions when initializing HRTF from a Filter object.')
            self.samplerate = data.samplerate
            fir = data.fir
            if fir:
                self.datatype = 'FIR'
            else:
                self.datatype = 'TF'
            # reshape the filterbank data to fit into HRTF (ind x taps x ear)
            data = data.data.T[..., None]
            self.data = []
            for idx in range(data.shape[0]):
                self.data.append(Filter(data[idx, :, :].T, self.samplerate, fir=fir))
            self.sources = HRTF._get_coordinates(sources, 'spherical')
            if listener is None:
                self.listener = [0, 0, 0]
            else:
                self.listener = listener
        elif isinstance(data, numpy.ndarray):
            if samplerate is None:  # should have shape (ind x ear x taps), 2 x n_taps filter (left right)
                raise ValueError('Must specify samplerate when initialising HRTF from an array.')
            self.samplerate = samplerate
            if datatype is None or type(datatype) is not str:
                raise ValueError('Must specify data type (FIR or TF) when initialising HRTF from an array.')
            self.datatype = datatype.upper()
            if self.datatype == 'FIR':
                fir = True
            elif self.datatype == 'TF':
                fir = False
            else:
                raise ValueError(f'Unsupported data type: {datatype}')
            if sources is None:
                raise ValueError('Must provide vertical-polar source positions when initializing HRTF from an array.')
            self.sources = HRTF._get_coordinates(sources, 'spherical')
            self.data = []
            for idx in range(data.shape[0]):
                self.data.append(Filter(data[idx, :, :].T, self.samplerate, fir=fir))
            if listener is None:
                self.listener = [0, 0, 0]
            else:
                self.listener = listener
        else:
            raise ValueError(f'Unsupported data type: {type(data)}')

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)} \n{repr(self.samplerate)})'

    def __str__(self):
        return f'{type(self)}, datatype {self.datatype}, sources {self.n_sources}, ' \
               f'elevations {self.n_elevations}, samplerate {self.samplerate}, samples {self[0].n_samples}'

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    # Static methods (used in __init__)
    @staticmethod
    def _sofa_load(filename, verbose=False):
        """
        Read a SOFA file.

        Arguments:
            f (h5netcdf.core.File): data as returned by the `_sofa_load` method.
        Returns:
            (h5netcdf.core.File): the data from the .sofa file.
        """
        if h5netcdf is False:
            raise ImportError('Reading from sofa files requires h5py and h5netcdf.')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = h5netcdf.File(filename, 'r')
        if verbose:
            f.items()
        return f

    @staticmethod
    def _sofa_get_data(f):
        """
        Read the impulse response or transfer functions from the SOFA data and store them in a Filter object.
        Return the Filter object in the `data` attribute, the data type and samplerate.

        Arguments:
            f (h5netcdf.core.File): data as returned by the `_sofa_load` method.
        Returns:
            data (list): a list of Filter objects containing the HRTF data.
            datatype (string): the data type used in the SOFA file
            samplerate (float): the sampling rate in Hz.
        """
        samplerate = HRTF._sofa_get_samplerate(f)
        datatype = f.attrs['DataType']
        data = []
        if datatype == 'FIR':
            ir_data = numpy.array(f.variables['Data.IR'], dtype='float')
            for idx in range(ir_data.shape[0]):
                data.append(Filter(ir_data[idx, :, :].T, samplerate))  # n_taps x 2 (left, right) filter
        elif datatype == 'TF':
            data_real = numpy.array(f.variables['Data.Real'], dtype='float')
            data_imag = numpy.array(f.variables['Data.Imag'], dtype='float')
            tf_data = numpy.abs(numpy.vectorize(complex)(data_real, data_imag))
            for idx in range(tf_data.shape[0]):
                data.append(Filter(tf_data[idx, :, :].T, samplerate, fir=False))
        else:
            raise NotImplementedError('Unsuppored datatype: {self.datatype}')
        return data, datatype, samplerate

    @staticmethod
    def _sofa_get_samplerate(f):
        """
        Returns the sampling rate of the recordings. If the sampling rate is not given in Hz, the function assumes
        it is given in kHz and multiplies by 1000 to convert to Hz.

        Arguments:
            f (h5netcdf.core.File): data as returned by the `_sofa_load` method.
        Returns:
            (float): the sampling rate in Hz.
        """
        attr = dict(f.variables['Data.SamplingRate'].attrs.items())  # get attributes as dict
        unit = attr['Units']  # extract and decode Units
        if unit in ('hertz', 'Hz'):
            return (numpy.array(f.variables['Data.SamplingRate'], dtype='int'))
        warnings.warn('Unit other than Hz. ' + unit + '. Assuming kHz.')
        return 1000 * (numpy.array(f.variables['Data.SamplingRate'], dtype='int'))

    @staticmethod
    def _sofa_get_sources(f):
        """
        Returns an array of positions of all sound sources.

        Arguments:
            f (h5netcdf.core.File): data as returned by the _sofa_load method.
        Returns:
            (numpy.ndarray): coordinates of all sources.
            (string): coordinate system used in the SOFA file.
        """
        # spherical coordinates, (azi,ele,radius), azi 0..360 (0=front, 90=left, 180=back), ele -90..90
        sources = numpy.array(f.variables['SourcePosition'], dtype='float')
        attr = dict(f.variables['SourcePosition'].attrs.items())  # get attributes as dict
        coordinate_system = attr['Type'].split(',')[0]  # extract and decode Units
        return sources, coordinate_system

    @staticmethod
    def _sofa_get_listener(f):
        """
        Returns dict with listeners positional information - used for plotting.

        Arguments:
            f (h5netcdf.core.File): data as returned by the `_sofa_load()` method.
        Returns:
            (dict): position of the listener ("pos"), the point which the listener is fixating ("view")
                the point 90° above the listener ("up") and vectors from the listener to those points.
        """
        lis = {'pos': numpy.array(f.variables['ListenerPosition'], dtype='float')[0],
               'view': numpy.array(f.variables['ListenerView'], dtype='float')[0],
               'up': numpy.array(f.variables['ListenerUp'], dtype='float')[0]}
        lis['viewvec'] = numpy.concatenate([lis['pos'], lis['pos']+lis['view']])
        lis['upvec'] = numpy.concatenate([lis['pos'], lis['pos']+lis['up']])
        return lis

    @staticmethod
    def _get_coordinates(sources, coordinate_system):
        """
        Returns the sound source positions in three different coordinate systems:
        cartesian, vertical-polar and interaural-polar.

        Arguments:
            sources (numpy.ndarray): sound source coordinates in cartesian coordinates (x, y, z),
            vertical-polar or interaural-polar coordinates (azimuth, elevation, distance).
            coordinate_system (string): type of the provided coordinates. Can be 'cartesian',
            'vertical_polar' or 'interaural_polar'.
        Returns:
            (named tuple): cartesian, vertical-polar and interaural-polar coordinates of all sources.
        """
        if isinstance(sources, (list, tuple)):
            sources = numpy.array(sources)
        if len(sources.shape) == 1:  # a single location (vector) needs to be converted to a 2d matrix
            sources = sources[numpy.newaxis, ...]
        sources = sources.astype('float64')
        source_coordinates = namedtuple('sources', 'cartesian vertical_polar interaural_polar')
        if coordinate_system == 'spherical':
            vertical_polar = sources
            cartesian = HRTF._vertical_polar_to_cartesian(vertical_polar)
            interaural_polar = HRTF._vertical_polar_to_interaural_polar(vertical_polar)
        elif coordinate_system == 'interaural':
            interaural_polar = sources
            cartesian = HRTF._interaural_polar_to_cartesian(interaural_polar)
            vertical_polar = HRTF._cartesian_to_vertical_polar(cartesian)
        elif coordinate_system == 'cartesian':
            cartesian = sources
            vertical_polar = HRTF._cartesian_to_vertical_polar(cartesian)
            interaural_polar = HRTF._vertical_polar_to_interaural_polar(vertical_polar)
        else:
            warnings.warn('Unrecognized coordinate system for source positions: ' + coordinate_system)
            return None
        return source_coordinates(cartesian.astype('float16'), vertical_polar.astype('float16'),
                                  interaural_polar.astype('float16'))

    @staticmethod
    def _vertical_polar_to_cartesian(vertical_polar):
        """
        Convert vertical-polar to cartesian coordinates.

        Arguments:
            vertical_polar (numpy.ndarray): vertical-polar coordinates (azimuth, elevation, distance).
        Returns:
            (numpy.ndarray): cartesian coordinates.
        """
        cartesian = numpy.zeros_like(vertical_polar)
        azimuths = numpy.deg2rad(vertical_polar[:, 0])
        elevations = numpy.deg2rad(90 - vertical_polar[:, 1])
        r = vertical_polar[:, 2].mean()  # get radii of sound sources
        cartesian[:, 0] = r * numpy.cos(azimuths) * numpy.sin(elevations)
        cartesian[:, 1] = r * numpy.sin(elevations) * numpy.sin(azimuths)
        cartesian[:, 2] = r * numpy.cos(elevations)
        return cartesian

    @staticmethod
    def _interaural_polar_to_cartesian(interaural_polar):
        """
        Convert interaural-polar to cartesian coordinates.

        Arguments:
            interaural_polar (numpy.ndarray): interaural-polar coordinates (azimuth, elevation, distance).
        Returns:
            (numpy.ndarray): cartesian coordinates.
        """
        cartesian = numpy.zeros_like(interaural_polar)
        azimuths = numpy.deg2rad(interaural_polar[:, 0])
        elevations = numpy.deg2rad(90 - interaural_polar[:, 1])
        r = interaural_polar[:, 2].mean()  # get radii of sound sources
        cartesian[:, 0] = r * numpy.cos(azimuths) * numpy.sin(elevations)
        cartesian[:, 1] = r * numpy.sin(azimuths)
        cartesian[:, 2] = r * numpy.cos(elevations) * numpy.cos(azimuths)
        return cartesian

    @staticmethod
    def _cartesian_to_vertical_polar(cartesian):
        """
        Convert cartesian to vertical-polar coordinates.

        Arguments:
            cartesian (numpy.ndarray): cartesian coordinates (azimuth, elevation, distance).
        Returns:
            (numpy.ndarray): vertical-polar coordinates.
        """
        vertical_polar = numpy.zeros_like(cartesian)
        xy = cartesian[:, 0] ** 2 + cartesian[:, 1] ** 2
        vertical_polar[:, 0] = numpy.rad2deg(numpy.arctan2(cartesian[:, 1], cartesian[:, 0]))
        vertical_polar[vertical_polar[:, 0] < 0, 0] += 360
        vertical_polar[:, 1] = 90 - numpy.rad2deg(numpy.arctan2(numpy.sqrt(xy), cartesian[:, 2]))
        vertical_polar[:, 2] = numpy.sqrt(xy + cartesian[:, 2] ** 2)
        return vertical_polar

    @staticmethod
    def _vertical_polar_to_interaural_polar(vertical_polar):
        """
        Convert vertical-polar to interaural-polar coordinates.

        Arguments:
            vertical_polar (numpy.ndarray): cartesian coordinates (azimuth, elevation, distance).
        Returns:
            (numpy.ndarray): interaural-polar coordinates.
        """
        interaural_polar = numpy.zeros_like(vertical_polar)
        azimuths = numpy.deg2rad(vertical_polar[:, 0])
        elevations = numpy.deg2rad(vertical_polar[:, 1])
        interaural_polar[:, 0] = numpy.rad2deg(numpy.arcsin(numpy.cos(elevations) * numpy.sin(azimuths)))
        with numpy.errstate(divide='ignore'):
            interaural_polar[:, 1] = (numpy.pi / 2) - numpy.arctan(((1 / numpy.tan(elevations)) * numpy.cos(azimuths)))
        interaural_polar[elevations < 0, 1] += numpy.pi
        interaural_polar[:, 1] = numpy.rad2deg(interaural_polar[:, 1])
        interaural_polar[:, 2] = vertical_polar[:, 2]
        return interaural_polar

    def apply(self, source, sound, allow_resampling=True):
        """
        Apply a filter from the HRTF set to a sound. The sound will be recast as slab.Binaural. If the samplerates
        of the sound and the HRTF are unequal and `allow_resampling` is True, then the sound will be resampled to the
        filter rate, filtered, and then resampled to the original rate.
        The filtering is done with `scipy.signal.fftconvolve`.

        Arguments:
            source (int): the source index of the binaural filter in self.data.
            sound (slab.Signal | slab.Sound | slab.Binaural): the sound to be rendered spatially.
        Returns:
            (slab.Binaural): a spatialized copy of `sound`.
        """
        from slab.binaural import Binaural  # importing here to avoid circular import at top of class
        if self.datatype == 'FIR':
            if (sound.samplerate != self.samplerate) and (not allow_resampling):
                raise ValueError('Filter and sound must have same sampling rates.')
            original_rate = sound.samplerate
            sound = sound.resample(self.samplerate)  # does nothing if samplerates are the same
            left = scipy.signal.fftconvolve(sound[:, 0], self[source][:, 0])
            if sound.n_channels == 1:
                right = scipy.signal.fftconvolve(sound[:, 0], self[source][:, 1])
            else:
                right = scipy.signal.fftconvolve(sound[:, 1], self[source][:, 1])
            convolved_sig = Signal([left, right], samplerate=self.samplerate)
            out = copy.deepcopy(sound)
            out.data = convolved_sig.data
            return Binaural(out.resample(original_rate))
        if self.datatype == 'TF':  # Filter.apply DTF as Fourier filter
            return self[source].apply(sound)

    def elevations(self):
        """
        Get all different elevations at which sources where recorded . Note: This currently only works as
        intended for HRTFs recorded in horizontal rings.

        Returns:
             (list): a sorted list of source elevations.
        """
        return sorted(list(set(numpy.round(self.sources.vertical_polar[:, 1]))))

    def plot_tf(self, sourceidx, ear='left', xlim=(1000, 18000), n_bins=None, kind='waterfall',
                linesep=20, xscale='linear', show=True, axis=None):
        """
        Plot transfer functions at a list of source indices.

        Arguments:
            ear (str): the ear from which data is plotted. Can be 'left', 'right', or 'both'.
            sourceidx (list of int): sources to plot. Typically be generated using the `hrtf.cone_sources` Method.
            xlim (tuple of int): frequency range of the plot
            n_bins (int) : passed to :meth:`slab.Filter.tf` and determines frequency resolution
            kind (str): type of plot to draw. Can be `waterfall` (as in Wightman and Kistler, 1989),
                `image` (as in Hofman, 1998) or 'surface' (as in Schnupp and Nelken, 2011).
            linesep (int): vertical distance between transfer functions in the waterfall plot
            xscale (str): sets x-axis scaling ('linear', 'log')
            show (bool): If True, show the plot immediately
            axis (matplotlib.axes._subplots.AxesSubplot): Axis to draw the plot on
        """
        if matplotlib is False:
            raise ImportError('Plotting HRTFs requires matplotlib.')
        if ear == 'left':
            chan = 0
        elif ear == 'right':
            chan = 1
        elif ear == 'both':
            if axis is not None and not isinstance(axis, (list, numpy.ndarray)):
                raise ValueError("Axis must be a list of length two when plotting left and right ear!")
            elif axis is None:
                axis = [None, None]
            if kind == 'image':
                fig1 = self.plot_tf(sourceidx, ear='left', xlim=xlim, axis=axis[0], show=show,
                                    linesep=linesep, n_bins=n_bins, kind='image', xscale=xscale)
                fig2 = self.plot_tf(sourceidx, ear='right', xlim=xlim, axis=axis[1], show=show,
                                    linesep=linesep, n_bins=n_bins, kind='image', xscale=xscale)
            elif kind == "waterfall":
                fig1 = self.plot_tf(sourceidx, ear='left', xlim=xlim, axis=axis[0], show=show,
                                    linesep=linesep, n_bins=n_bins, kind='waterfall', xscale=xscale)
                fig2 = self.plot_tf(sourceidx, ear='right', xlim=xlim, axis=axis[1], show=show,
                                    linesep=linesep, n_bins=n_bins, kind='waterfall', xscale=xscale)
            else:
                raise ValueError("'Kind' must be either 'waterfall' or 'image'!")
            return fig1, fig2
        else:
            raise ValueError("Unknown value for ear. Use 'left', 'right', or 'both'")
        if not axis:
            fig, axis = plt.subplots()
        else:
            fig = axis.figure
        if kind == 'waterfall':
            vlines = numpy.arange(0, len(sourceidx)) * linesep
            for idx, source in enumerate(sourceidx):
                filt = self[source]
                freqs, h = filt.tf(channels=chan, n_bins=n_bins, show=False)
                axis.plot(freqs, h + vlines[idx],
                          linewidth=0.75, color='0.0', alpha=0.7)
            ticks = vlines[::2]  # plot every second elevation
            labels = numpy.round(self.sources.vertical_polar[sourceidx, 1]*2, decimals=-1)/2
            # plot every third elevation label, omit comma to save space
            labels = labels[::2].astype(int)
            axis.set(yticks=ticks, yticklabels=labels)
            axis.grid(visible=True, axis='y', which='both', linewidth=0.25)
            axis.plot([xlim[0]+500, xlim[0]+500], [vlines[-1]+10, vlines[-1] +
                      10+linesep], linewidth=1, color='0.0', alpha=0.9)
            axis.text(x=xlim[0]+600, y=vlines[-1]+10+linesep/2,
                      s=str(linesep)+'dB', va='center', ha='left', fontsize=6, alpha=0.7)
        elif kind == 'image' or 'surface':
            if not n_bins:
                img = numpy.zeros((self[sourceidx[0]].n_taps, len(sourceidx)))
            else:
                img = numpy.zeros((n_bins, len(sourceidx)))
            elevations = self.sources.vertical_polar[sourceidx, 1]
            for idx, source in enumerate(sourceidx):
                filt = self[source]
                freqs, h = filt.tf(channels=chan, n_bins=n_bins, show=False)
                img[:, idx] = h.flatten()
            img[img < -25] = -25  # clip at -40 dB transfer
            if kind == 'image':
                contour = axis.contourf(freqs[freqs <= xlim[1]], elevations, img.T[:, freqs <= xlim[1]], cmap='hot', origin='upper', levels=20)
                divider = make_axes_locatable(axis)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(contour, cax, orientation="vertical")
            elif kind == 'surface':
                xi, yi = numpy.meshgrid(freqs, elevations)  # interpolate to smooth surface plot
                spline = scipy.interpolate.Rbf(xi, yi, img.T, function='thin_plate')  # interpolator instance
                x, y = numpy.meshgrid(numpy.linspace(freqs.min(), freqs.max(), len(freqs)),
                                      numpy.linspace(elevations.min(), elevations.max(), 100))
                z = spline(x, y)
                x[x < xlim[0]] = numpy.nan  # trim edges
                x[x > xlim[1]] = numpy.nan
                fig, axis = plt.subplots()
                axis.axis('off')
                axis = plt.axes(projection='3d')
                contour = axis.plot_surface(x, y, z, rcount=200, ccount=200, cmap='cool')
                fig.colorbar(contour, fraction=0.046, pad=0.04, orientation="horizontal")
        else:
            raise ValueError("Unknown plot type. Use 'waterfall' or 'image'.")
        axis.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x / 1000))))
        axis.autoscale(tight=True)
        axis.tick_params('both', length=2, pad=2)
        if kind == 'surface':
            axis.set(xlabel='Frequency [Hz]', ylabel='Elevation [˚]', zlabel='Pinna gain [dB]',
                     xlim=xlim, xscale=xscale)
        else:
            axis.set(xlabel='Frequency [Hz]', ylabel='Elevation [˚]', xlim=xlim,
                     xscale=xscale)
        if show:
            plt.show()

    def diffuse_field_avg(self):
        """
        Compute the diffuse field average transfer function, i.e. the constant non-spatial portion of a set of HRTFs.
        The filters for all sources are averaged, which yields an unbiased average only if the sources are uniformly
        distributed around the head.

        Returns:
             (Filter): the diffuse field average as FFR filter object.
        """
        dfa = []
        for source in range(self.n_sources):
            filt = self[source]
            for chan in range(filt.n_channels):
                _, h = filt.tf(channels=chan, show=False)
                dfa.append(h)
        dfa = 10 ** (numpy.mean(dfa, axis=0)/20)  # average and convert from dB to gain
        return Filter(dfa, fir=False, samplerate=self.samplerate)

    def diffuse_field_equalization(self, dfa=None):
        """
        Equalize the HRTF by dividing each filter by the diffuse field average. The resulting filters have a mean
        close to 0 and are Fourier filters.

        Arguments:
            dfa (None): Filter object containing the diffuse field average transfer function of the HRTF.
                If none is provided, the `diffuse_field_avg` method is called to obtain it.
        Returns:
            (HRTF): diffuse field equalized version of the HRTF.
        """
        if dfa is None:
            dfa = self.diffuse_field_avg()
        # invert the diffuse field average
        dfa.data = 1/dfa.data
        dtfs = copy.deepcopy(self)
        # apply the inverted filter to the HRTFs
        for source in range(dtfs.n_sources):
            filt = dtfs.data[source]
            _, h = filt.tf(show=False)
            h = 10 ** (h / 20) * dfa
            dtfs.data[source] = Filter(data=h, fir=False, samplerate=self.samplerate)
        return dtfs

    def cone_sources(self, cone=0, full_cone=False):
        """
        Get all sources of the HRTF that lie on a "cone of confusion". The cone is a vertical off-axis sphere
        slice. All sources that lie on the cone have the same interaural level and time difference.
        Note: This currently only works as intended for HRTFs recorded in horizontal rings.

        Arguments:
            cone (int | float): azimuth of the cone center in degree.
            full_cone (bool): If True, return all sources that lie on the cone, otherwise, return only sources
                in front of the listener.
        Returns:
            (list): elements of the list are the indices of sound sources on the frontal half of the cone.
        Examples::

            import HRTF
            hrtf = slab.HRTF.kemar()
            sourceidx = hrtf.cone_sources(20)  # get the source indices
            print(hrtf.sources[sourceidx])  # print the coordinates of the source indices
            hrtf.plot_sources(sourceidx)  # show the sources in a 3D plot
        """
        cone = numpy.sin(numpy.deg2rad(cone))
        eles = self.elevations()
        _cartesian = self.sources.cartesian / 1.4  # get cartesian coordinates on the unit sphere
        out = []
        for ele in eles:  # for each elevation, find the source closest to the reference y
            if full_cone == False:  # only return cone sources in front of listener
                subidx, = numpy.where((numpy.round(self.sources.vertical_polar[:, 1]) == ele) & (_cartesian[:, 0] >= 0))
            else:  # include cone sources behind listener
                subidx, = numpy.where(numpy.round(self.sources.vertical_polar[:, 1]) == ele)
            cmin = numpy.min(numpy.abs(_cartesian[subidx, 1] - cone).astype('float16'))
            if cmin < 0.05:  # only include elevation where the closest source is less than 5 cm away
                idx, = numpy.where((numpy.round(self.sources.vertical_polar[:, 1]) == ele) & (
                        numpy.abs(_cartesian[:, 1] - cone).astype('float16') == cmin))  # avoid rounding error
                out.append(idx[0])
                if full_cone and len(idx) > 1:
                    out.append(idx[1])
        return sorted(out, key=lambda x: self.sources.vertical_polar[x, 1])

    def elevation_sources(self, elevation=0):
        """
        Get the indices of sources along a horizontal sphere slice at the given `elevation`.

        Arguments:
            elevation (int | float): The elevation of the sources in degree. The default returns sources along
                the frontal horizon.
        Returns:
            (list): indices of the sound sources. If the hrtf does not contain the specified `elevation` an empty
                list is returned.
        """
        idx = numpy.where((self.sources.vertical_polar[:, 1] == elevation) & (
                (self.sources.vertical_polar[:, 0] <= 90) | (self.sources.vertical_polar[:, 0] >= 270)))
        return idx[0].tolist()

    def tfs_from_sources(self, sources, n_bins=96):
        """
        Get the transfer function from sources in the hrtf.

        Arguments:
            sources (list): Indices of the sources (as generated for instance with the `HRTF.cone_sources` method),
                for which the transfer function is extracted.
            n_bins (int): The number of frequency bins for each transfer function.
        Returns:
            (numpy.ndarray): 2-dimensional array where the first dimension represents the frequency bins and the
                second dimension represents the sources.
        """
        n_sources = len(sources)
        tfs = numpy.zeros((n_bins, n_sources))
        for idx, source in enumerate(sources):
            _, jwd = self[source].tf(channels=0, n_bins=n_bins, show=False)
            tfs[:, idx] = jwd.flatten()
        return tfs

    def interpolate(self, azimuth=0, elevation=0, method='nearest', plot_tri=False):
        """
        Interpolate a filter at a given azimuth and elevation from the neighboring HRTFs. A weighted average of the
        3 closest HRTFs in the set is computed in the spectral domain with barycentric weights. The resulting filter
        values vary smoothly with changes in azimuth and elevation. The fidelity of the interpolated filter decreases
        with increasing distance of the closest sources and should only be regarded as appropriate approximation when
        the contributing filters are less than 20˚ away.

        Arguments:
            azimuth (float): the azimuth component of the direction of the interpolated filter
            elevation (float): the elevation component of the direction of the interpolated filter
            method (str): interpolation method, 'nearest' returns the filter of the nearest direction. Any other string
                returns a barycentric interpolation.
            plot_tri (bool): plot the triangulation of source positions used of interpolation. Useful for checking
                for areas where the interpolation may not be accurate (look for irregular or elongated triangles).
        Returns:
            (slab.HRTF): an HRTF object with a single source
        """
        from slab.binaural import Binaural  # importing here to avoid circular import at top of class
        coordinates = self.sources.cartesian
        r = self.sources.vertical_polar[:, 2].mean()
        target = self._get_coordinates((azimuth, elevation, r), 'spherical').cartesian
        # compute distances from target direction
        distances = numpy.sqrt(((target - coordinates)**2).sum(axis=1))
        if method == 'nearest':
            idx_nearest = numpy.argmin(distances)
            filt = self[idx_nearest]
        else:
            # triangulate source positions into triangles
            if not scipy:
                raise ImportError('Need scipy.spatial for barycentric interpolation.')
            tri = scipy.spatial.ConvexHull(coordinates)
            if plot_tri:
                ax = plt.subplot(projection='3d')
                for simplex in tri.points[tri.simplices]:
                    polygon = Poly3DCollection([simplex])
                    polygon.set_color(numpy.random.rand(3))
                    ax.add_collection3d(polygon)
                    mins = coordinates.min(axis=0)
                    maxs = coordinates.max(axis=0)
                    xlim, ylim, zlim = list(zip(mins, maxs))
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.set_zlim(zlim)
                    ax.set_xlabel('X [m]')
                    ax.set_ylabel('Y [m]')
                    ax.set_zlabel('Z [m]')
                    plt.show()
            # for each simplex, find the coordinates, test if target in triangle (by finding minimal d)
            d_min = numpy.inf
            for i, vertex_list in enumerate(tri.simplices):
                simplex = tri.points[vertex_list]
                d, a = HRTF._barycentric_weights(simplex, target)
                if d < d_min:
                    d_min, idx, weights = d, i, a
            vertex_list = tri.simplices[idx]
            # we now have the indices of the filters and the corresponding weights
            amplitudes = list()
            for idx in vertex_list:
                freqs, amps = self[idx].tf(show=False)  # get their transfer functions
                amplitudes.append(amps)  # we could interpolate here if frequencies differ between filters
            avg_amps = amplitudes[0] * weights[0] + amplitudes[1] * weights[1] + amplitudes[2] * weights[2]  # average
            gains = avg_amps - avg_amps.max()  # shift so that maximum is zero, because we can only attenuate
            gains[gains < -60] = -60  # limit dynamic range to 60 dB
            gains_lin = 10**(gains/20)  # transform attenuations in dB to factors
            filt_l = Filter.band(frequency=list(freqs), gain=list(gains_lin[:, 0]), length=self[idx].n_samples, fir=True,
                                 samplerate=self[vertex_list[0]].samplerate)
            filt_r = Filter.band(frequency=list(freqs), gain=list(gains_lin[:, 1]), length=self[idx].n_samples, fir=True,
                                 samplerate=self[vertex_list[0]].samplerate)
            filt = Filter(data=[filt_l, filt_r])
            itds = list()
            for idx in vertex_list:
                taps = Binaural(self[idx])  # recast filter taps as Binaural sound
                itds.append(taps.itd())  # use Binaural.itd to compute correlation lag between channels
            avg_itd = itds[0] * weights[0] + itds[1] * weights[1] + itds[2] * weights[2]  # average ITD
            filt = filt.delay(avg_itd / self.samplerate)
        data = filt.data[numpy.newaxis, ...]  # get into correct shape (idx, taps, ear)
        source_loc = numpy.array([[azimuth, elevation, r]])
        out = HRTF(data, datatype='FIR', samplerate=self.samplerate, sources=source_loc, listener=self.listener)
        return out

    @staticmethod
    def _barycentric_weights(triangle, point):
        '''
        Returns:
            (None | numpy.array): barycentric weights for a given triangle (array of coordinates of points) and target
            point IF the point is inside the triangle; None if the point is outside the triangle.
        '''
        # compute barycentric weights via the area of the 3 triangles formed between target and 3 nearest sources:
        dist = lambda p1, p2: numpy.sqrt(((p1 - p2)**2).sum())
        d1 = dist(point, triangle[0, :])  # distances from target to each source
        d2 = dist(point, triangle[1, :])
        d3 = dist(point, triangle[2, :])
        d12 = dist(triangle[0, :], triangle[1, :])  # distance between sources 1 and 2
        d13 = dist(triangle[0, :], triangle[2, :])
        d23 = dist(triangle[1, :], triangle[2, :])
        # compute triangle areas from length of sides (distances) with Heron's Formula
        a = numpy.array([0., 0., 0.])
        p = (d2 + d3 + d23) / 2
        a[0] = numpy.sqrt(p * (p-d2) * (p-d3) * (p-d23))
        p = (d1 + d3 + d13) / 2
        a[1] = numpy.sqrt(p * (p-d1) * (p-d3) * (p-d13))
        p = (d1 + d2 + d12) / 2
        a[2] = numpy.sqrt(p * (p-d1) * (p-d2) * (p-d12))
        p = (d12 + d13 + d23) / 2
        tot = numpy.sqrt(p * (p-d12) * (p-d13) * (p-d23))
        return a.sum() - tot, a / a.sum()  # normalize by total area = barycentric weights of sources in idx_triangle

    def vsi(self, sources=None, equalize=True):
        """
        Compute  the "vertical spectral information" which is a measure of the dissimilarity of spectral profiles
        at different elevations. The vsi relates to behavioral localization accuracy in the vertical dimension
        (Trapeau and Schönwiesner, 2016). It is computed as one minus the average of the correlation coefficients
        between all combinations of directional transfer functions of the specified `sources`. A set of identical
        transfer functions results in a vsi of 0 whereas highly different transfer functions will result in a high VSI
        (empirical maximum is ~1.07, KEMAR has a VSI of 0.82).

        Arguments:
            sources (None | list): indices of sources for which to compute the VSI. If None use the vertical midline.
            equalize (bool): If True, apply the `diffuse_field_equalization` method (set to False if the hrtf object
                is already diffuse-field equalized).
        Returns:
            (float): the vertical spectral information between the specified `sources`.
        """
        if sources is None:
            sources = self.cone_sources()
        if equalize:
            dtf = self.diffuse_field_equalization()
            tfs = dtf.tfs_from_sources(sources=sources)
        else:
            tfs = self.tfs_from_sources(sources=sources)
        sum_corr = 0
        n = 0
        for i in range(len(sources)):
            for j in range(i+1, len(sources)):
                sum_corr += numpy.corrcoef(tfs[:, i], tfs[:, j])[1, 0]
                n += 1
        return 1 - sum_corr / n

    def plot_sources(self, idx=None, show=True, label=False, axis=None):
        """
        Plot source locations in 3D.

        Arguments:
            idx (list of int): indices to highlight in the plot
            show (bool): whether to show plot (set to False if plotting into an axis and you want to add other elements)
            label (bool): if True, show the index of each source in self.sources as text label, if idx is also given,
                then only theses sources are labeled
            axis (mpl_toolkits.mplot3d.axes3d.Axes3D): axis to draw the plot on
        """
        if matplotlib is False or Axes3D is False:
            raise ImportError('Plotting 3D sources requires matplotlib and mpl_toolkits')
        if axis is None:
            ax = plt.subplot(projection='3d')
        else:
            if not isinstance(axis, Axes3D):
                raise ValueError("Axis must be instance of Axes3D!")
            ax = axis
        coordinates = self.sources.cartesian
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='.')
        ax.axes.set_xlim3d(left=numpy.min(coordinates), right=numpy.max(coordinates))
        if label and idx is None:
            for i in range(coordinates.shape[0]):
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'{i}', size=8, zorder=1, color='k')
        ax.scatter(0, 0, 0, c='r', marker='o')
        if self.listener:
            x_, y_, z_, u, v, w = zip(*[self.listener['viewvec'], self.listener['upvec']])
            ax.quiver(x_, y_, z_, u, v, w, length=0.5, colors=['r', 'b', 'r', 'r', 'b', 'b'])
        if idx is not None:
            ax.scatter(coordinates[idx, 0], coordinates[idx, 1], coordinates[idx, 2], c='r', marker='o')
            if label:
                for i in idx:
                    ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], f'{i}', size=8, zorder=1, color='k')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        if show:
            plt.show()

    @staticmethod
    def kemar():
        '''
        Provides HRTF data from the KEMAR recording (normal pinna) conducted by Gardner and Martin at MIT in 1994
        (MIT Media Lab Perceptual Computing - Technical Report #280) and converted to the SOFA Format. Slab includes a
        compressed copy of the data. This function reads it and returns the corresponding HRTF object. The objects is
        cached in the class variable `_kemar` and repeated calls return the cached object instead of reading the file
        from disk again.

        Returns:
            (slab.HRTF): the KEMAR HRTF data.
        '''
        global _kemar
        if _kemar is None:
            kemar_path = pathlib.Path(__file__).parent.resolve() / pathlib.Path('data') / 'mit_kemar_normal_pinna.bz2'
            _kemar = pickle.load(bz2.BZ2File(kemar_path, "r"))
            _kemar.sources = HRTF._get_coordinates(_kemar.sources, 'spherical')
            _kemar.datatype = 'FIR'
        return _kemar

    @staticmethod
    def estimate_hrtf(recordings, signal, sources, listener=None):
        """
        Compute a set of transfer functions from binaural recordings and an input (reference) signal.
        For each sound source, compute the DFT of left- and right-ear recordings
        and divide by the Fourier transform of the input signal to obtain the head related transfer function.

        Arguments:
            signal (slab.Signal | slab.Sound): the signal used to produce the in-ear recordings.
            recordings (list): in-ear recordings stored in a list of slab.Binaural objects.
            sources (numpy.array): interaural polar coordinates (azimuth, elevation, distance) of all sources,
                number and order of sources must match the recordings.

        Returns:
            (slab.HRTF): an HRTF object with the dimensions specified by the recordings and the source file.
        """
        if not isinstance(recordings, list) or not isinstance(recordings[0], Sound):
            raise ValueError('Recordings must be provided as a list of slab.Sound objects.')
        if isinstance(sources, (list, tuple)):
            sources = numpy.array(sources)
        if len(sources.shape) == 1:  # a single location (vector) needs to be converted to a 2d matrix
            sources = sources[numpy.newaxis, ...]
        if len(sources) != len(recordings):
            raise ValueError('Number of sound sources must be equal to number of recordings.')
        rec_samplerate = recordings[0].samplerate
        rec_n_samples = recordings[0].n_samples
        rec_data = []
        for recording in recordings:
            if not (recording.n_channels == 2 and recording.n_samples == recordings[0].n_samples
                    and recording.samplerate == rec_samplerate):
                raise ValueError('Number of channels, samples and samplerate must be equal for all recordings.')
            rec = copy.deepcopy(recording)
            # rec.data -= numpy.mean(rec.data, axis=0)  # remove DC component in FFT of recordings
            rec_data.append(rec.data.T)
        rec_data = numpy.asarray(rec_data)
        if not signal.samplerate == rec_samplerate:
            signal = signal.resample(rec_samplerate)
        if not signal.n_samples == rec_n_samples:
            sig_freq_bins = numpy.fft.rfftfreq(signal.n_samples, d=1 / signal.samplerate)
            rec_freq_bins = numpy.fft.rfftfreq(rec_n_samples, d=1 / rec_samplerate)
            sig_fft = numpy.interp(rec_freq_bins, sig_freq_bins, numpy.fft.rfft(signal.data[:, 0]))
        else:
            sig_fft = numpy.fft.rfft(signal.data[:, 0])
        with numpy.errstate(divide='ignore'):
            hrtf_data = numpy.fft.rfft(rec_data) / sig_fft
        if not listener:
            listener = {'pos': numpy.array([0., 0., 0.]), 'view': numpy.array([1., 0., 0.]),
                        'up': numpy.array([0., 0., 1.]), 'viewvec': numpy.array([0., 0., 0., 1., 0., 0.]),
                        'upvec': numpy.array([0., 0., 0., 0., 0., 1.])}
        return HRTF(data=numpy.abs(hrtf_data), datatype='TF', samplerate=rec_samplerate, sources=sources,
                    listener=listener)

    def write_sofa(self, filename):
        """
        Save the HRTF data to a SOFA file.

        Arguments:
            filename (str | pathlib.Path): path, the file is written to.
        """
        if isinstance(filename, pathlib.Path):
            filename = str(filename)
        if pathlib.Path(filename).is_file():
            pathlib.Path(filename).unlink()  # overwrite if filename already exists
        if netCDF4 is False:
            raise ImportError('Writing sofa files requires netCDF4.')
        sofa = netCDF4.Dataset(filename, 'w', format='NETCDF4')  # Create SOFA file
        # ----------Dimensions----------#
        m = self.n_sources  # number of measurements (= n_sources)
        n = self[0].n_samples  # n_samples - frequencies of fourier filter or taps of FIR filter
        r = 2  # number of receivers (HRTFs measured for 2 ears)
        e = 1  # number of emitters (1 speaker per measurement)
        i = 1  # always 1
        c = 3  # number of dimensions in space (elevation, azimuth, radius)
        sofa.createDimension('M', m)
        sofa.createDimension('N', n)
        sofa.createDimension('E', e)
        sofa.createDimension('R', r)
        sofa.createDimension('I', i)
        sofa.createDimension('C', c)
        # ----------Attributes----------#
        sofa.DataType = self.datatype
        if self.datatype == 'FIR':
            sofa.SOFAConventions, sofa.SOFAConventionsVersion = 'SimpleFreeFieldHRIR', '2.0'
            delayVar = sofa.createVariable('Data.Delay', 'f8', ('I', 'R'))
            delay = numpy.zeros((i, r))
            delayVar[:, :] = delay
            dataIRVar = sofa.createVariable('Data.IR', 'f8', ('M', 'R', 'N'))
            dataIRVar.ChannelOrdering = 'acn'
            dataIRVar.Normalization = 'sn3d'
            IR_data = []
            for idx in numpy.asarray(self[:]):
                IR_data.append(idx.T)
            dataIRVar[:] = numpy.asarray(IR_data)
        elif self.datatype == 'TF':
            sofa.SOFAConventions, sofa.SOFAConventionsVersion = 'SimpleFreeFieldHRTF', '2.0'
            dataRealVar = sofa.createVariable('Data.Real', 'f8', ('M', 'R', 'N'))  # data
            TF_data = []
            for idx in numpy.asarray(self[:]):
                TF_data.append(idx.T)
            dataRealVar[:] = numpy.asarray(TF_data)
            dataImagVar = sofa.createVariable('Data.Imag', 'f8', ('M', 'R', 'N'))
            dataImagVar[:] = numpy.zeros((m, r, n))  # for internal use, store real data only
            NVar = sofa.createVariable('N', 'f8', ('N'))
            NVar.LongName = 'frequency'
            NVar.Units = 'hertz'
            NVar[:] = n
        sofa.RoomType = 'free field'
        sofa.Conventions, sofa.Version = 'SOFA', '2.0'
        sofa.APIName, sofa.APIVersion = 'pysofaconventions', '0.1'
        sofa.AuthorContact, sofa.License = 'Leipzig University', 'PublicLicence'
        sofa.ListenerShortName, sofa.Organization = 'sub01', 'Eurecat - UPF'
        sofa.DateCreated, sofa.DateModified = str(datetime.datetime.now()), str(datetime.datetime.now())
        sofa.Title, sofa.DatabaseName = 'sofa_title', 'UniLeipzig Freefield'
        # ----------Variables----------#
        listenerPositionVar = sofa.createVariable('ListenerPosition', 'f8', ('I', 'C'))
        listenerPositionVar.Units = 'metre'
        listenerPositionVar.Type = 'cartesian'
        listenerPositionVar[:] = self.listener['pos']
        receiverPositionVar = sofa.createVariable('ReceiverPosition', 'f8', ('R', 'C', 'I'))
        receiverPositionVar.Units = 'metre'
        receiverPositionVar.Type = 'cartesian'
        receiverPositionVar[:] = numpy.zeros((r, c, i))
        sourcePositionVar = sofa.createVariable('SourcePosition', 'f8', ('M', 'C'))
        sourcePositionVar.Units = 'degree, degree, metre'
        sourcePositionVar.Type = 'spherical'
        sourcePositionVar[:] = self.sources.vertical_polar  # array of speaker positions
        emitterPositionVar = sofa.createVariable('EmitterPosition', 'f8', ('E', 'C', 'I'))
        emitterPositionVar.Units = 'metre'
        emitterPositionVar.Type = 'cartesian'
        emitterPositionVar[:] = numpy.zeros((e, c, i))
        listenerUpVar = sofa.createVariable('ListenerUp', 'f8', ('I', 'C'))
        listenerUpVar.Units = 'metre'
        listenerUpVar.Type = 'cartesian'
        listenerUpVar[:] = self.listener['up']
        listenerViewVar = sofa.createVariable('ListenerView', 'f8', ('I', 'C'))
        listenerViewVar.Units = 'metre'
        listenerViewVar.Type = 'cartesian'
        listenerViewVar[:] = self.listener['view']
        samplingRateVar = sofa.createVariable('Data.SamplingRate', 'f8', ('I'))
        samplingRateVar.Units = 'hertz'
        samplingRateVar[:] = self.samplerate
        sofa.close()
