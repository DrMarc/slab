'''
Class for reading and manipulating head-related transfer functions. Reads files in .sofa format (started before
python implementations of the sofa conventions were available -> will be migrated to use pysofaconventions!)
'''

import copy
import warnings
import pathlib
import numpy

try:
    import matplotlib
    import matplotlib.pyplot as plt
    have_pyplot = True
except ImportError:
    have_pyplot = False
try:
    from mpl_toolkits.mplot3d import Axes3D
    have_mplot3d = True
except ImportError:
    have_mplot3d = False
try:
    import h5netcdf
    have_h5 = True
except ImportError:
    have_h5 = False

from slab.filter import Filter


class HRTF():
    '''
    Class for reading and manipulating head-related transfer functions. This is essentially a collection of two Filter
    objects (hrtf.left and hrtf.right) with attributes (`nsources`, `nelevations`) and functions to manage them.

    >>> hrtf = HRTF(data='mit_kemar_normal_pinna.sofa') # initialize from sofa file
    >>> print(hrtf)
    <class 'hrtf.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0
    >>> sourceidx = hrtf.cone_sources(20)
    >>> hrtf.plot_sources(sourceidx)
    >>> hrtf.plot_tf(sourceidx,ear='left')
    '''
    # instance properties
    nsources = property(fget=lambda self: len(self.sources),
                        doc='The number of sources in the HRTF.')
    nelevations = property(fget=lambda self: len(self.elevations()),
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
                # ntaps x 2 (left, right) filter
                self.data.append(Filter(data[idx, :, :].T, self.samplerate))
            self.listener = HRTF._sofa_get_listener(f)
            self.sources = HRTF._sofa_get_sourcepositions(f)
        elif isinstance(data,
         Filter):
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
            self.samplerate = samplerate
            self.data = []
            for idx in range(data.shape[0]):
                # (ind x taps x ear), 2 x ntaps filter (left right)
                self.data.append(Filter(data[idx, :, :].T, self.samplerate))
            self.sources = sources
            self.listener = listener

    def __repr__(self):
        return f'{type(self)} (\n{repr(self.data)} \n{repr(self.samplerate)})'

    def __str__(self):
        return f'{type(self)} sources {self.nsources}, elevations {self.nelevations}, samples {self.data[0].nsamples}, samplerate {self.samplerate}'

    # Static methods (used in __init__)
    @staticmethod
    def _sofa_load(filename, verbose=False):
        'Reads a SOFA file and returns a h5netcdf structure'
        if not have_h5:
            raise ImportError('Reading from sofa files requires h5py and h5netcdf.')
        f = h5netcdf.File(filename, 'r')
        if verbose:
            f.items()
        return f

    @staticmethod
    def _sofa_get_samplerate(f):
        'Returns the sampling rate of the recordings'
        attr = dict(f.variables['Data.SamplingRate'].attrs.items())  # get attributes as dict
        unit = attr['Units'].decode('UTF-8')  # extract and decode Units
        if unit in ('hertz', 'Hz'):
            return float(numpy.array(f.variables['Data.SamplingRate'], dtype='float'))
        warnings.warn('Unit other than Hz. ' + unit + '. Assuming kHz.')
        return 1000 * float(numpy.array(f.variables['Data.SamplingRate'], dtype='float'))

    @staticmethod
    def _sofa_get_sourcepositions(f):
        'Returns an array of positions of all sound sources'
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
        '''Returns dict with listener attributes from a sofa file handle; keys: pos, view, up, viewvec, upvec.
        Used for adding a listener vector in plot functions.'''
        lis = {}
        lis['pos'] = numpy.array(f.variables['ListenerPosition'], dtype='float')[0]
        lis['view'] = numpy.array(f.variables['ListenerView'], dtype='float')[0]
        lis['up'] = numpy.array(f.variables['ListenerUp'], dtype='float')[0]
        lis['viewvec'] = numpy.concatenate([lis['pos'], lis['pos']+lis['view']])
        lis['upvec'] = numpy.concatenate([lis['pos'], lis['pos']+lis['up']])
        return lis

    @staticmethod
    def _sofa_get_FIR(f):
        'Returns an array of FIR filters for all source positions from a sofa file handle.'
        datatype = f.attrs['DataType'].decode('UTF-8')  # get data type
        if datatype != 'FIR':
            warnings.warn('Non-FIR data: ' + datatype)
        return numpy.array(f.variables['Data.IR'], dtype='float')

    # instance methods
    def elevations(self):
        '''Return the list of sources.
        Note: This currently only works as intended for HRTFs recorded in horizontal rings.
        '''
        return sorted(list(set(numpy.round(self.sources[:, 1]))))

    def plot_tf(self, sourceidx, ear='left', xlim=(1000, 18000), nbins=None, kind='waterfall',
                linesep=20, xscale='linear', show=True, axis=None):
        """
        Plots transfer functions of FIR filters at a list of source indices.
        Args:
            ear (str): which ear to plot ('left', 'right', 'both').
            sourceidx (list of int): sources to plot. should typically be generated with
                                    `hrtf.cone_sources(cone=0)` for midline sources at all elevations
            xlim (tuple of int): frequency range of the plot
            nbins (int) : passed to :meth:`slab.Filter.tf` and determines freqency resolution
            kind (str): `waterfall` (as in Wightman and Kistler, 1989) and
                        `image` plots (as in Hofman 1998) are available
            linesep (int): vertical distance between transfer functions in the waterfall plot
            xscale (str): sets x-axis scaling ('linear', 'log')
            show (bool): Whether to show plot or
            axis (matplotlib.axes._subplots.AxesSubplot): Axis to draw the plot on
        """
        if not have_pyplot:
            raise ImportError('Plotting HRTFs requires matplotlib.')
        if ear == 'left':
            chan = 0
        elif ear == 'right':
            chan = 1
        elif ear == 'both':
            chan = [0, 1]
            if kind == 'image':
                fig1 = self.plot_tf(sourceidx, ear='left', xlim=xlim,
                                    linesep=linesep, nbins=nbins, kind='image', xscale=xscale)
                fig2 = self.plot_tf(sourceidx, ear='right', xlim=xlim,
                                    linesep=linesep, nbins=nbins, kind='image', xscale=xscale)
                return fig1, fig2
        else:
            raise ValueError("Unknown value for ear. Use 'left', 'right', or 'both'")
        if not axis:
            fig, axis = plt.subplots()
        if kind == 'waterfall':
            vlines = numpy.arange(0, len(sourceidx)) * linesep
            for idx, s in enumerate(sourceidx):
                filt = self.data[s]
                freqs, h = filt.tf(channels=chan, nbins=nbins, show=False)
                axis.plot(freqs, h + vlines[idx],
                          linewidth=0.75, color='0.0', alpha=0.7)
            ticks = vlines[::3]  # plot every third elevation
            labels = numpy.round(self.sources[sourceidx, 1]*2, decimals=-1)/2
            # plot every third elevation label, ommit comma to save space
            labels = labels[::3].astype(int)
            axis.set(yticks=ticks, yticklabels=labels)
            axis.grid(b=True, axis='y', which='both', linewidth=0.25)
            axis.plot([xlim[0]+500, xlim[0]+500], [vlines[-1]+10, vlines[-1] +
                                                   10+linesep], linewidth=1, color='0.0', alpha=0.9)
            axis.text(x=xlim[0]+600, y=vlines[-1]+10+linesep/2,
                      s=str(linesep)+'dB', va='center', ha='left', fontsize=6, alpha=0.7)
        elif kind == 'image':
            if not nbins:
                img = numpy.zeros((self.data[sourceidx[0]].ntaps, len(sourceidx)))
            else:
                img = numpy.zeros((nbins, len(sourceidx)))
            elevations = self.sources[sourceidx, 1]
            for idx, source in enumerate(sourceidx):
                filt = self.data[source]
                freqs, h = filt.tf(channels=chan, nbins=nbins, show=False)
                img[:, idx] = h.flatten()
            img[img < -25] = -25  # clip at -40 dB transfer
            plt.contourf(freqs, elevations, img.T, cmap='hot', origin='upper', levels=20)
            plt.colorbar()
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
        '''
        Compute the diffuse field average transfer function, i.e. the constant non-spatial portion of a set of HRTFs.
        The filters for all sources are averaged, which yields an unbiased average only if the sources are uniformely
        distributed around the head. Returns the diffuse field average as FFR filter object.
        '''  # TODO: could make the contribution of each HRTF depend on local density of sources.
        dfa = []
        for source in range(self.nsources):
            filt = self.data[source]
            for chan in range(filt.nchannels):
                _, h = filt.tf(channels=chan, show=False)
                dfa.append(h)
        dfa = 10 ** (numpy.mean(dfa, axis=0)/20)  # average and convert from dB to gain
        return Filter(dfa, fir=False, samplerate=self.samplerate)

    def diffuse_field_equalization(self):
        '''
        Returns a diffuse field equalized version of the HRTF. The resulting filters have zero mean and are of type FFR.
        '''
        dfa = self.diffuse_field_avg()
        # invert the diffuse field average
        dfa.data = 1/dfa.data
        dtfs = copy.deepcopy(self)
        # apply the inverted filter to the HRTFs
        for source in range(dtfs.nsources):
            filt = dtfs.data[source]
            _, h = filt.tf(show=False)
            h = 10 ** (h / 20) * dfa
            dtfs.data[source] = Filter(data=h, fir=False, samplerate=self.samplerate)
        return dtfs

    def cone_sources(self, cone=0):
        '''
        Return indices of sources along a vertical off-axis sphere slice (`cone`). The default returns sources along the
        fronal median plane. Note: This currently only works as intended for HRTFs recorded in horizontal rings.
        '''
        cone = numpy.sin(numpy.deg2rad(cone))
        azimuth = numpy.deg2rad(self.sources[:, 0])
        elevation = numpy.deg2rad(self.sources[:, 1]-90)
        x = numpy.sin(elevation) * numpy.cos(azimuth)
        y = numpy.sin(elevation) * numpy.sin(azimuth)
        eles = self.elevations()
        out = []
        for ele in eles:  # for each elevation, find the source closest to the target y
            subidx, = numpy.where((numpy.round(self.sources[:, 1]) == ele) & (x >= 0))
            cmin = numpy.min(numpy.abs(y[subidx]-cone))
            if cmin < 0.05:  # only include elevation where the closest source is less than 5 cm away
                idx, = numpy.where((numpy.round(self.sources[:, 1]) == ele) & (
                    numpy.abs(y-cone) == cmin))
                out.append(idx[0])
        return sorted(out, key=lambda x: self.sources[x, 1])

    def elevation_sources(self, elevation=0):
        '''
        Return indices of sources along a horizontal sphere slice at the given `elevation`.
        The default returns sources along the fronal horizon.
        '''
        idx = numpy.where((self.sources[:, 1] == elevation) & (
            (self.sources[:, 0] <= 90) | (self.sources[:, 0] >= 270)))
        return idx[0]

    def tfs_from_sources(self, source_list, n_bins=96):
        '''
        Extract transfer functions with `n_bins` from a list of source indices (`source_list`, generated for instance
        with :meth:`slab.HRTF.cone_sources`) as `(n_bins, n_sources)` numpy array.
        '''
        n_sources = len(source_list)
        tfs = numpy.zeros((n_bins, n_sources))
        for idx, source in enumerate(source_list):
            _, jwd = self.data[source].tf(channels=0, nbins=96, show=False)
            tfs[:, idx] = jwd.flatten()
        return tfs

    def vsi(self, sources=None, equalize=True):
        '''
        Compute a measure of the dissimilarity of spectral profiles at different elevations ("vertical spectral information"), which relates to behavioral localization accuracy in the vertical dimension (Trapeau and Schönwiesner, 2016).

        If `equalize` is True, the method applies a `diffuse_field_equalization()` (set to False if the hrtf object is already diffuse-field equalized), then computes the average of the correlation coefficients between all combinations of DTFs on the vertical midline (obtained from `cone_sources()`) or any other set of source indices supplied as `sources`. VSI measures the average dissimilarity of DTFs as 1 minus the average of the coefficients. A DTF set of identical transfer functions for all elevations will result in a VSI of 0, whereas highly different transfer functions will result in a high VSI (empirical maximum is ~1.07, KEMAR has a VSI of 0.82). To obtain the VSI measure from the paper, sources should on the vertical midline and the hrtf should be diffuse-field equalized.

        Attributes:
            sources: indices of sources for which to compute the VSI (default is the vertical midline)
            equalize: if True, apply diffuse field equalization, saves a bit of computing time when set to false
                      and the hrtf is already equalized.
        '''
        if sources is None:
            sources = self.cone_sources()
        if equalize:
            dtf = self.diffuse_field_equalization()
            tfs = dtf.tfs_from_sources(source_list=sources)
        else:
            tfs = self.tfs_from_sources(source_list=sources)
        sum_corr = 0
        n = 0
        for i in range(len(sources)):
            for j in range(i+1, len(sources)):
                sum_corr += numpy.corrcoef(tfs[:,i], tfs[:,j])[1,0]
                n += 1
        return 1 - sum_corr / n

    def plot_sources(self, idx=False, show=True, axis=None):
        """
        Plot source locations in 3D.
        Args:
            idx (list of int): Indices to highlight in den plot
            show (bool): Whether to show plot or
            axis (mpl_toolkits.mplot3d.axes3d.Axes3D): Axis to draw the plot on
        """
        if not have_pyplot and not have_mplot3d:
            raise ImportError('Plotting 3D sources requires matplotlib and mpl_toolkits')
        if axis in None:
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
        if idx:
            ax.scatter(x[idx], y[idx], z[idx], c='r', marker='o')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        if show:
            plt.show()


if __name__ == '__main__':
    from slab import DATAPATH
    hrtf = HRTF(data=DATAPATH+'mit_kemar_normal_pinna.sofa')
    hrtf.data[20].tf(plot=True)
