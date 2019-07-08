'''
Class for reading and manipulating head-related transfer functions.
'''

import warnings
import pathlib
import numpy

try:
	import matplotlib.pyplot as plt
	have_pyplot = True
except ImportError:
	have_pyplot = False
try:
	import scipy.signal
	have_scipy = True
except ImportError:
	have_scipy = False
try:
	import h5py
	import h5netcdf
	have_h5 = True
except ImportError:
	have_h5 = False

from slab.filter import Filter

class HRTF():
	'''
	Class for reading and manipulating head-related transfer functions. This is essentially
	a collection of two Filter objects (hrtf.left and hrtf.right) with functions to manage them.
	>>> hrtf = HRTF(data='mit_kemar_normal_pinna.sofa') # initialize from sofa file
	>>> print(hrtf)
	<class 'hrtf.HRTF'> sources 710, elevations 14, samples 710, samplerate 44100.0
	>>> sourceidx = hrtf.cone_sources(20)
	>>> hrtf.plot_sources(sourceidx)
	>>> hrtf.plot_tf(sourceidx,ear='left')

	'''
	# instance properties
	nsources = property(fget=lambda self:len(self.sources),
						doc='The number of sources in the HRTF.')
	nelevations = property(fget=lambda self:len(self.elevations()),
						doc='The number of elevations in the HRTF.')

	def __init__(self, data, samplerate=None, sources=None, listener=None, verbose=False):
		if isinstance(data, str):
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising HRTF from a file.')
			if pathlib.Path(data).suffix != '.sofa':
				raise NotImplementedError('Only .sofa files can be read at the moment.')
			else: # load from SOFA file
				try:
					f = HRTF._sofa_load(data, verbose)
				except:
					raise ValueError('Unable to read file.')
				data = HRTF._sofa_get_FIR(f)
				self.samplerate = HRTF._sofa_get_samplerate(f)
				self.data = []
				for idx in range(data.shape[0]):
					self.data.append(Filter(data[idx,:,:].T, self.samplerate)) # ntaps x 2 (left, right) filter
				self.listener = HRTF._sofa_get_listener(f)
				self.sources = HRTF._sofa_get_sourcepositions(f)
		else:
			self.samplerate = samplerate
			for idx in range(data.shape[0]):
				self.data.append(Filter(data[idx,:,:].T, self.samplerate)) # 2 x ntaps filter (left right)
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
		'returns the sampling rate of the recordings'
		attr = dict(f.variables['Data.SamplingRate'].attrs.items()) # get attributes as dict
		if attr['Units'].decode('UTF-8') == 'hertz': # extract and decode Units
			return float(numpy.array(f.variables['Data.SamplingRate'], dtype='float'))
		else: # Khz?
			warnings.warn('Unit other than Hz. ' + attr['Units'].decode('UTF-8') + '. Assuming kHz.')
			return 1000 * float(numpy.array(f.variables['Data.SamplingRate'], dtype='float'))

	@staticmethod
	def _sofa_get_sourcepositions(f):
		'returns an array of positions of all sound sources'
		# spherical coordinates, (azi,ele,radius), azi 0..360 (0=front, 90=left, 180=back), ele -90..90
		attr = dict(f.variables['SourcePosition'].attrs.items()) # get attributes as dict
		unit = attr['Units'].decode('UTF-8').split(',')[0] # extract and decode Units
		if unit in ('degree', 'degrees', 'deg'):
			return numpy.array(f.variables['SourcePosition'], dtype='float')
		elif unit in ('meter', 'meters', 'm'):
			# convert to azimuth and elevation
			sources = numpy.array(f.variables['SourcePosition'], dtype='float')
			x, y, z = sources[:,0], sources[:,1], sources[:,2]
			r = numpy.sqrt(x**2 + y**2 + z**2)
			azimuth = numpy.rad2deg(numpy.arctan2(y, x))
			elevation = 90 - numpy.rad2deg(numpy.arccos(z / r))
			return numpy.stack((azimuth, elevation, r), axis=1)
		else:
			warnings.warn('Unrecognized unit for source positions: ' + unit)
			return numpy.array(f.variables['SourcePosition'], dtype='float') # fall back to no conversion

	@staticmethod
	def _sofa_get_listener(f):
		'''Returns dict with listener attributes from a sofa file handle.
		Keys: pos, view, up, viewvec, upvec. Used for adding a listener vector in plot functions.'''
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
		datatype = f.attrs['DataType'].decode('UTF-8') # get data type
		if datatype != 'FIR':
			warnings.warn('Non-FIR data: ' + datatype)
		return numpy.array(f.variables['Data.IR'], dtype='float')

	# instance methods
	def elevations(self):
		'Return the list of sources'
		return sorted(list(set(self.sources[:,1])))

	def plot_tf(self, sourceidx, ear='left', linesep=20, nbins=None, kind='waterfall'):
		'''
		Plots transfer functions of FIR filters for a given ear
		['left', 'right', 'both'] at a list of source indices.
		Sourceidx should be generated like this: hrtf.cone_sources(cone=0).
		Waterfall (as in Wightman and Kistler, 1989) and image plots
		(as in Hofman 1998) are available.
		'''
		n = 0
		if ear == 'left':
			chan = 0
		elif ear == 'right':
			chan = 1
		elif ear == 'both':
			chan = [0, 1]
			if kind == 'image':
				fig1 = self.plot_tf(sourceidx, ear='left', linesep=linesep, nbins=nbins, kind='image')
				fig2 = self.plot_tf(sourceidx, ear='right', linesep=linesep, nbins=nbins, kind='image')
				return fig1, fig2
		else:
			raise ValueError("Unknown value for ear. Use 'left', 'right', or 'both'")
		if kind == 'waterfall':
			fig = plt.figure()
			for s in sourceidx:
				filt = self.data[s]
				freqs, h = filt.tf(channels=chan, nbins=nbins, plot=False)
				plt.plot(freqs, h + n, label=str(self.sources[s, 1])+'˚')
				n += linesep
			plt.ylabel('Amplitude [dB]') # TODO: should be a calibration bar
			plt.grid()
		elif kind == 'image':
			if not nbins:
				img = numpy.zeros((self.data[sourceidx[0]].ntaps, len(sourceidx)))
			else:
				img = numpy.zeros((nbins, len(sourceidx)))
			elevations = self.sources[sourceidx, 1]
			for idx, source in enumerate(sourceidx):
				filt = self.data[source]
				freqs, h = filt.tf(channels=chan, nbins=nbins, plot=False)
				img[:,idx] = h.flatten()
			img[img < -25] = -25 # clip at -40 dB transfer
			fig = plt.figure()
			plt.contourf(freqs, elevations, img.T, cmap='hot', origin='upper', levels=20)
			#plt.xscale('log')
			plt.colorbar()
			plt.ylabel('Elevation [˚]') # TODO: missing colorbar for amplitude
		else:
			raise ValueError("Unknown plot type. Use 'waterfall' or 'image'.")
		plt.xlabel('Frequency [kHz]')
		plt.autoscale(tight=True)
		plt.xlim([1000, 16000])
		plt.xscale('log')
		plt.show()
		return fig

	def diffuse_field_avg(self):
		'''
		Compute the diffuse field average transfer function,
		i.e. the constant non-spatial portion of a set of HRTFs.
		The filters for all sources are averaged, which yields
		an unbiased average only if the sources are uniformely
		distributed around the head.
		Returns the diffuse field average as FFR filter object.
		''' # TODO: could make the contribution of each HRTF
		# depend on local density of sources.
		dfa = []
		for source in range(self.nsources):
			filt = self.data[source]
			for chan in range(filt.nchannels):
				_, h = filt.tf(channels=chan, plot=False)
				dfa.append(h)
		dfa = 10 ** (numpy.mean(dfa, axis=0)/20) # average and convert from dB to gain
		return Filter(dfa, fir=False, samplerate=self.samplerate)

	def diffuse_field_equalization(self):
		'''
		Apply a diffuse field equalization to an HRTF in place.
		The resulting filters have zero mean and are of type FFR.
		'''
		dfa = self.diffuse_field_avg()
		# invert the diffuse field average
		dfa.data = 1/dfa.data
		# apply the inverted filter to the HRTFs
		for source in range(self.nsources):
			filt = self.data[source]
			_, h = filt.tf(plot=False)
			h = 10 ** (h / 20) * dfa
			self.data[source] = Filter(data=h, fir=False, samplerate=self.samplerate)

	def cone_sources(self, cone=0):
		'''
		Return indices of sources along an off-axis sphere slice.
		The default cone = 0 returns sources along the fronal median plane.
		'''
		cone = numpy.sin(numpy.deg2rad(cone))
		azimuth = numpy.deg2rad(self.sources[:,0])
		elevation = numpy.deg2rad(self.sources[:,1]-90)
		x = numpy.sin(elevation) * numpy.cos(azimuth)
		y = numpy.sin(elevation) * numpy.sin(azimuth)
		eles = self.elevations()
		out = []
		for ele in eles: # for each elevation, find the source closest to the target y
			subidx, = numpy.where((self.sources[:,1] == ele) & (x >= 0))
			cmin = numpy.min(numpy.abs(y[subidx]-cone))
			idx, = numpy.where((self.sources[:,1] == ele) & (numpy.abs(y-cone) == cmin))
			out.append(idx[0])
		return sorted(out, key=lambda x: self.sources[x,1])

	def elevation_sources(self, elevation=0):
		'Return indices of sources along an off-axis sphere slice.'
		idx = numpy.where((hrtf.sources[:,1] == elevation) & ((hrtf.sources[:,0] <= 90) | (hrtf.sources[:,0] >= 270)))
		return idx[0]

	def plot_sources(self, idx=False):
		'DOC'
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = Axes3D(fig)
		azimuth = numpy.deg2rad(self.sources[:,0])
		elevation = numpy.deg2rad(self.sources[:,1]-90)
		r = self.sources[:,2]
		x = r * numpy.sin(elevation) * numpy.cos(azimuth)
		y = r * numpy.sin(elevation) * numpy.sin(azimuth)
		z = r * numpy.cos(elevation)
		ax.scatter(x,y,z, c ='b', marker='.')
		ax.scatter(0,0,0, c ='r', marker='o')
		if self.listener: # TODO: view dir is inverted!
			x_, y_, z_, u, v, w = zip(*[self.listener['viewvec'],self.listener['upvec']])
			ax.quiver(x_, y_, z_, u, v, w, length=0.5, colors=['r','b','r','r','b','b'])
		if idx:
			ax.scatter(x[idx], y[idx], z[idx], c='r', marker='o')
		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		ax.set_zlabel('Z [m]')
		plt.show()

if __name__ == '__main__':
	from slab import DATAPATH
	hrtf = HRTF(data=DATAPATH+'mit_kemar_normal_pinna.sofa')
	hrtf.data[20].tf(plot=True)
