"""
Class for reading and manipulating head-related transfer functions. Reads files in .sofa format (started before
python implementations of the sofa conventions were available -> will be migrated to use pysofaconventions!)
"""

import copy
import warnings
import pathlib
import numpy
try:
    import matplotlib
    from matplotlib import pyplot as plt
except ImportError:
    matplotlib, plt = False, False
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    Axes3D = False
    make_axes_locatable = False
try:
    import h5netcdf
except ImportError:
    h5netcdf = False
from slab.filter import Filter


class HRTF:
    """ Class for reading and manipulating head-related transfer functions with attributes and functions to manage them.
    Arguments:
        data (str | Filter | numpy.ndarray): Typically, this is the path to a file in the .sofa format.
            The file is then loaded and the data of each source for which the transfer function was recorded is stored
            as a Filter object in the `data` attribute. Instead of a file name, the data can be passed directly as
            Filter or numpy array (not recommended). Given a `Filter`, every filter channel in the instance is taken as
            a sound source. Given an 3-dimensional array, the first dimension represents the sources, the second the
            number of taps per filter and the last the number of filter channels per filter (should be always 2 for
            left and right ear).
        samplerate (None | float): rate at which the data was acquired, only relevant when not loading from .sofa file
        sources (None | array): positions of the recorded sources, only relevant when not loading from .sofa file
        listener (None | list | dict): position of the listener, only relevant when not loading from .sofa file
        verbose (bool): print out items when loading .sofa files, defaults to False
    Attributes:
        .n_sources (int): The number of sources in the HRTF.
        .sources (array): spherical coordinates (azimuth, elevation, distance) of all sources.
        .n_elevations (int): The number of elevations in the HRTF.
        . data (list): The HRTF data. The elements of the list are instances of slab.Filter.
        .listener (dict): a dictionary containing the position of the listener ("pos"), the point which the listener
            is fixating ("view"), the point 90° above the listener ("up") and vectors from the listener to those points.
        .samplerate (float): sampling rate at which the HRTF data was acquired.
    Example:
        from slab import DATAPATH, HRTF
        hrtf = slab.HRTF(data=DATAPATH+'mit_kemar_normal_pinna.sofa')  # initialize from sofa file
        sourceidx = hrtf.cone_sources(20)
        hrtf.plot_sources(sourceidx)
        hrtf.plot_tf(sourceidx,ear='left') """
    # instance properties
    n_sources = property(fget=lambda self: len(self.sources),
                         doc='The number of sources in the HRTF.')
    n_elevations = property(fget=lambda self: len(self.elevations()),
                            doc='The number of elevations in the HRTF.')

    def __init__(self, data, samplerate=None, sources=None, listener=None, verbose=False):
        if isinstance(data, str):
            if samplerate is not None:
                raise ValueError('Cannot specify samplerate when initialising HRTF from a file.')
            if pathlib.Path(data).suffix != '.sofa':
                raise NotImplementedError('Only .sofa files can be read.')
            f = HRTF._sofa_load(data, verbose)
            data = HRTF._sofa_get_FIR(f)
            self.samplerate = HRTF._sofa_get_samplerate(f)
            self.data = []
            for idx in range(data.shape[0]):
                # n_taps x 2 (left, right) filter
                self.data.append(Filter(data[idx, :, :].T, self.samplerate))
            self.listener = HRTF._sofa_get_listener(f)
            self.sources = HRTF._sofa_get_sourcepositions(f)
        elif isinstance(data, Filter):
            # This is a hacky shortcut for casting a filterbank as HRTF. Avoid unless you know what you are doing.
            if sources is None:
                raise ValueError('Must provide source positions when using a Filter object.')
            self.samplerate = data.samplerate
            fir = data.fir  # save the fir property of the filterbank
            # reshape the filterbank data to fit into HRTF (ind x taps x ear)
            data = data.data.T[..., None]
            self.data = []
            for idx in range(data.shape[0]):
                self.data.append(Filter(data[idx, :, :].T, self.samplerate, fir=fir))
            self.sources = sources
            if listener is None:
                self.listener = [0, 0, 0]
            else:
                self.listener = listener
        else:
            self.samplerate = samplerate
            self.data = []
            for idx in range(data.shape[0]):
                # (ind x taps x ear), 2 x n_taps filter (left right)
                self.data.append(Filter(data[idx, :, :].T, self.samplerate))
            self.sources = sources
            if listener is None:
                self.listener = [0, 0, 0]
            else:
                self.listener = listener

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)} \n{repr(self.samplerate)})'

    def __str__(self):
        return f'{type(self)} sources {self.n_sources}, elevations {self.n_elevations},' \
               f'samples {self.data[0].nsamples}, samplerate {self.samplerate}'

    # Static methods (used in __init__)
    @staticmethod
    def _sofa_load(filename, verbose=False):
        """ Read a SOFA file.
        Arguments:
            filename (str): full path to the .sofa file.
        Returns:
            (h5netcdf.core.File): the data from the .sofa file. """
        if h5netcdf is False:
            raise ImportError('Reading from sofa files requires h5py and h5netcdf.')
        f = h5netcdf.File(filename, 'r')
        if verbose:
            f.items()
        return f

    @staticmethod
    def _sofa_get_samplerate(f):
        """ Returns the sampling rate of the recordings. If the sampling rate is not given in Hz, the function assumes
        it is given in kHz and multiplies by 1000 to convert to Hz.
        Arguments:
            f (h5netcdf.core.File): data as returned by the `_sofa_load` method.
        Returns:
            (float): the sampling rate in Hz.
        """
        attr = dict(f.variables['Data.SamplingRate'].attrs.items())  # get attributes as dict
        unit = attr['Units'].decode('UTF-8')  # extract and decode Units
        if unit in ('hertz', 'Hz'):
            return float(numpy.array(f.variables['Data.SamplingRate'], dtype='float'))
        warnings.warn('Unit other than Hz. ' + unit + '. Assuming kHz.')
        return 1000 * float(numpy.array(f.variables['Data.SamplingRate'], dtype='float'))

    @staticmethod
    def _sofa_get_sourcepositions(f):
        """ Returns an array of positions of all sound sources.
        Arguments:
            f (h5netcdf.core.File): data as returned by the _sofa_load method.
        Returns:
            (numpy.ndarray): spherical coordinates (azimuth, elevation, distance) of all sources.
        """
        # spherical coordinates, (azi,ele,radius), azi 0..360 (0=front, 90=left, 180=back), ele -90..90
        attr = dict(f.variables['SourcePosition'].attrs.items())  # get attributes as dict
        unit = attr['Units'].decode('UTF-8').split(',')[0]  # extract and decode Units
        if unit in ('degree', 'degrees', 'deg'):
            return numpy.array(f.variables['SourcePosition'], dtype='float')
        if unit in ('meter', 'meters', 'metre', 'metres', 'm'):
            # convert to azimuth and elevation
            sources = numpy.array(f.variables['SourcePosition'], dtype='float')
            x, y, z = sources[:, 0], sources[:, 1], sources[:, 2]
            r = numpy.sqrt(x**2 + y**2 + z**2)
            azimuth = numpy.rad2deg(numpy.arctan2(y, x))
            elevation = 90 - numpy.rad2deg(numpy.arccos(z / r))
            return numpy.stack((azimuth, elevation, r), axis=1)
        warnings.warn('Unrecognized unit for source positions: ' + unit)
        # fall back to no conversion
        return numpy.array(f.variables['SourcePosition'], dtype='float')

    @staticmethod
    def _sofa_get_listener(f):
        """ Returns dict with listeners positional information - used for plotting.
        Attributes:
            f (h5netcdf.core.File): data as returned by the `_sofa_load()` method.
        Returns:
            (dict): position of the listener ("pos"), the point which the listener is fixating ("view")
                the point 90° above the listener ("up") and vectors from the listener to those points. """
        lis = {'pos': numpy.array(f.variables['ListenerPosition'], dtype='float')[0],
               'view': numpy.array(f.variables['ListenerView'], dtype='float')[0],
               'up': numpy.array(f.variables['ListenerUp'], dtype='float')[0]}
        lis['viewvec'] = numpy.concatenate([lis['pos'], lis['pos']+lis['view']])
        lis['upvec'] = numpy.concatenate([lis['pos'], lis['pos']+lis['up']])
        return lis

    @staticmethod
    def _sofa_get_FIR(f):
        """ Returns an array of FIR filters for all source positions.
        Attributes:
            f (h5netcdf.core.File): data as returned by the `_sofa_load()` method.
        Returns:
            (numpy.ndarray): a 3-dimensional array where the first dimension represents the number of sources from
                which data was recorded and the second dimension represents the left and right ear. """
        datatype = f.attrs['DataType'].decode('UTF-8')  # get data type
        if datatype != 'FIR':
            warnings.warn('Non-FIR data: ' + datatype)
        return numpy.array(f.variables['Data.IR'], dtype='float')

    def elevations(self):
        """ Get all different elevations at which sources where recorded . Note: This currently only works as
        intended for HRTFs recorded in horizontal rings.
         Returns:
             (list): a sorted list of source elevations."""
        return sorted(list(set(numpy.round(self.sources[:, 1]))))

    def plot_tf(self, sourceidx, ear='left', xlim=(1000, 18000), n_bins=None, kind='waterfall',
                linesep=20, xscale='linear', show=True, axis=None):
        """ Plot transfer functions of FIR filters at a list of source indices.
        Arguments:
            ear (str): the ear from which data is plotted. Can be 'left', 'right', or 'both'.
            sourceidx (list of int): sources to plot. Typically be generated using the `hrtf.cone_sources` Method.
            xlim (tuple of int): frequency range of the plot
            n_bins (int) : passed to :meth:`slab.Filter.tf` and determines frequency resolution
            kind (str): type of plot to draw. Can be `waterfall` (as in Wightman and Kistler, 1989) or
                `image` (as in Hofman 1998).
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
            for idx, s in enumerate(sourceidx):
                filt = self.data[s]
                freqs, h = filt.tf(channels=chan, nbins=n_bins, show=False)
                axis.plot(freqs, h + vlines[idx],
                          linewidth=0.75, color='0.0', alpha=0.7)
            ticks = vlines[::3]  # plot every third elevation
            labels = numpy.round(self.sources[sourceidx, 1]*2, decimals=-1)/2
            # plot every third elevation label, omit comma to save space
            labels = labels[::3].astype(int)
            axis.set(yticks=ticks, yticklabels=labels)
            axis.grid(b=True, axis='y', which='both', linewidth=0.25)
            axis.plot([xlim[0]+500, xlim[0]+500], [vlines[-1]+10, vlines[-1] +
                                                   10+linesep], linewidth=1, color='0.0', alpha=0.9)
            axis.text(x=xlim[0]+600, y=vlines[-1]+10+linesep/2,
                      s=str(linesep)+'dB', va='center', ha='left', fontsize=6, alpha=0.7)
        elif kind == 'image':
            if not n_bins:
                img = numpy.zeros((self.data[sourceidx[0]].n_taps, len(sourceidx)))
            else:
                img = numpy.zeros((n_bins, len(sourceidx)))
            elevations = self.sources[sourceidx, 1]
            for idx, source in enumerate(sourceidx):
                filt = self.data[source]
                freqs, h = filt.tf(channels=chan, nbins=n_bins, show=False)
                img[:, idx] = h.flatten()
            img[img < -25] = -25  # clip at -40 dB transfer
            contour = axis.contourf(freqs, elevations, img.T, cmap='hot', origin='upper', levels=20)
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(contour, cax, orientation="vertical")

        else:
            raise ValueError("Unknown plot type. Use 'waterfall' or 'image'.")
        axis.autoscale(tight=True)
        axis.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, pos: str(int(x/1000))))
        axis.tick_params('both', length=2, pad=2)
        axis.set(xlabel='Frequency [kHz]', ylabel='Elevation [˚]', xlim=xlim, xscale=xscale)
        if show:
            plt.show()

    def diffuse_field_avg(self):
        """
        Compute the diffuse field average transfer function, i.e. the constant non-spatial portion of a set of HRTFs.
        The filters for all sources are averaged, which yields an unbiased average only if the sources are uniformly
        distributed around the head.
        Returns:
             (Filter): the diffuse field average as FFR filter object. """
        # TODO: could make the contribution of each HRTF depend on local density of sources.
        dfa = []
        for source in range(self.n_sources):
            filt = self.data[source]
            for chan in range(filt.n_channels):
                _, h = filt.tf(channels=chan, show=False)
                dfa.append(h)
        dfa = 10 ** (numpy.mean(dfa, axis=0)/20)  # average and convert from dB to gain
        return Filter(dfa, fir=False, samplerate=self.samplerate)

    def diffuse_field_equalization(self):
        """ Equalize the HRTF by dividing each filter by the diffuse field average. The resulting filters have a mean
        close to 0 and they are Fourier filters.
        Returns:
            (HRTF): diffuse field equalized version of the HRTF. """
        # TODO: the filter mean is not 0 after equalization
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

    def cone_sources(self, cone=0):
        """ Get all sources of the HRTF that lie on a "cone of confusion". The cone is a vertical off-axis sphere
        slice. All sources that lie on the cone have the same interaural level and time difference.
        Note: This currently only works as intended for HRTFs recorded in horizontal rings.
        Arguments:
            cone (int | float): azimuth of the cone center in degree.
        Returns:
            (list): elements of the list are the indices of sound sources on the frontal half of the cone.
        Examples:
            from slab import DATAPATH, HRTF
            hrtf = slab.HRTF(data=DATAPATH+'mit_kemar_normal_pinna.sofa')  # initialize from sofa file
            sourceidx = hrtf.cone_sources(20)  # get the source indices
            print(hrtf.sources[sourceidx])  # print the coordinates of the source indices
            hrtf.plot_sources(sourceidx)  # show the sources in a 3D plot """
        cone = numpy.sin(numpy.deg2rad(cone))
        azimuth = numpy.deg2rad(self.sources[:, 0])
        elevation = numpy.deg2rad(self.sources[:, 1]-90)
        # the points defined by x and y are the source locations projected onto the azimuth plane
        x = numpy.sin(elevation) * numpy.cos(azimuth)
        y = numpy.sin(elevation) * numpy.sin(azimuth)
        eles = self.elevations()
        out = []
        for ele in eles:  # for each elevation, find the source closest to the reference y
            subidx, = numpy.where((numpy.round(self.sources[:, 1]) == ele) & (x >= 0))
            cmin = numpy.min(numpy.abs(y[subidx]-cone))
            if cmin < 0.05:  # only include elevation where the closest source is less than 5 cm away
                idx, = numpy.where((numpy.round(self.sources[:, 1]) == ele) & (
                    numpy.abs(y-cone) == cmin))
                out.append(idx[0])
        return sorted(out, key=lambda x: self.sources[x, 1])

    def elevation_sources(self, elevation=0):
        """ Get the indices of sources along a horizontal sphere slice at the given `elevation`.
        Arguments:
            elevation (int | float): The elevation of the sources in degree. The default returns sources along
                the frontal horizon.
        Returns:
            (list): indices of the sound sources. If the hrtf does not contain the specified `elevation` an empty
                list is returned. """
        idx = numpy.where((self.sources[:, 1] == elevation) & (
            (self.sources[:, 0] <= 90) | (self.sources[:, 0] >= 270)))
        return idx[0]

    def tfs_from_sources(self, sources, n_bins=96):
        """Get the transfer function from sources in the hrtf.
        Arguments:
            sources (list): Indices of the sources (as generated for instance with the `HRTF.cone_sources` method), for
                which the transfer function is extracted.
            n_bins (int): The number of frequency bins for each transfer function
        Returns:
            (numpy.ndarray): 2-dimensional array where the first dimension represents the frequency bins and the
                second dimension represents the sources. """
        n_sources = len(sources)
        tfs = numpy.zeros((n_bins, n_sources))
        for idx, source in enumerate(sources):
            _, jwd = self.data[source].tf(channels=0, n_bins=n_bins, show=False)
            tfs[:, idx] = jwd.flatten()
        return tfs

    def vsi(self, sources=None, equalize=True):
        """ Compute  the "vertical spectral information" which is a measure of the dissimilarity of spectral profiles
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
            (float): the vertical spectral information between the specified `sources`. """
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

    def plot_sources(self, idx=None, show=True, axis=None):
        """
        Plot source locations in 3D.
        Args:
            idx (list of int): Indices to highlight in den plot
            show (bool): Whether to show plot or
            axis (mpl_toolkits.mplot3d.axes3d.Axes3D): Axis to draw the plot on
        """
        if matplotlib is False or Axes3D is False:
            raise ImportError('Plotting 3D sources requires matplotlib and mpl_toolkits')
        if axis is None:
            ax = Axes3D(plt.figure())
        else:
            if not (isinstance(axis, Axes3D)):
                raise ValueError("Axis must be instance of Axes3D!")
            else:
                ax = axis
        azimuth = numpy.deg2rad(self.sources[:, 0])
        elevation = numpy.deg2rad(self.sources[:, 1]-90)
        r = self.sources[:, 2]
        x = r * numpy.sin(elevation) * numpy.cos(azimuth)
        y = r * numpy.sin(elevation) * numpy.sin(azimuth)
        z = r * numpy.cos(elevation)
        ax.scatter(x, y, z, c='b', marker='.')
        ax.scatter(0, 0, 0, c='r', marker='o')
        if self.listener:  # TODO: view dir is inverted!
            x_, y_, z_, u, v, w = zip(*[self.listener['viewvec'], self.listener['upvec']])
            ax.quiver(x_, y_, z_, u, v, w, length=0.5, colors=['r', 'b', 'r', 'r', 'b', 'b'])
        if idx is not None:
            ax.scatter(x[idx], y[idx], z[idx], c='r', marker='o')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        if show:
            plt.show()
