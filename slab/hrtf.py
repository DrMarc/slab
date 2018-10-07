'''
Class for reading and manipulating head-related transfer functions.
'''

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
import warnings
import pathlib

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

	def __init__(self,data,samplerate=None,sources=None,listener=None,verbose=False):
		if isinstance(data, str):
			if samplerate is not None:
				raise ValueError('Cannot specify samplerate when initialising HRTF from a file.')
			if pathlib.Path(data).suffix != '.sofa':
				raise NotImplementedError('Only .sofa files can be read at the moment.')
			else: # load from SOFA file
				try:
					f = HRTF._sofa_load(data,verbose)
				except:
					raise ValueError('Unable to read file.')
				data = HRTF._sofa_get_FIR(f)
				self.samplerate = HRTF._sofa_get_samplerate(f)
				self.left = Filter(data[:,0,:],self.samplerate) # create a Filter object for left ear data
				self.right = Filter(data[:,1,:],self.samplerate) # create a Filter object for right ear data
				self.listener = HRTF._sofa_get_listener(f)
				self.sources = HRTF._sofa_get_sourcepositions(f)
		else:
			self.samplerate = samplerate
			self.left = Filter(data[:,0,:],self.samplerate)
			self.right = Filter(data[:,1,:],self.samplerate)
			self.sources = sources
			self.listener = listener

	def __repr__(self):
		return f'{type(self)} (\n{repr(self.left.data)} \n{repr(self.right.data)} \n{repr(self.samplerate)})'

	def __str__(self):
		return f'{type(self)} sources {self.nsources}, elevations {self.nelevations}, samples {self.left.nsamples}, samplerate {self.samplerate}'

	# Static methods (used in __init__)
	@staticmethod
	def _sofa_load(filename,verbose=False):
		'Reads a SOFA file and returns a h5netcdf structure'
		if not have_h5:
			raise ImportError('Reading from sofa files requires h5py and h5netcdf.')
		f = h5netcdf.File(filename,'r')
		if verbose:
			f.items()
		return f

	@staticmethod
	def _sofa_get_samplerate(f):
		'returns the sampling rate of the recordings'
		attr = dict(f.variables['Data.SamplingRate'].attrs.items()) # get attributes as dict
		if attr['Units'].decode('UTF-8') == 'hertz': # extract and decode Units
			return float(numpy.array(f.variables['Data.SamplingRate'],dtype='float'))
		else: # Khz?
			warnings.warn('Unit other than Hz. ' + attr['Units'].decode('UTF-8') + '. Assuming kHz.')
			return 1000 * float(numpy.array(f.variables['Data.SamplingRate'],dtype='float'))

	@staticmethod
	def _sofa_get_sourcepositions(f):
		'returns an array of positions of all sound sources'
		# spherical coordinates, (azi,ele,radius), azi 0..360 (0=front, 90=left, 180=back), ele -90..90
		attr = dict(f.variables['SourcePosition'].attrs.items()) # get attributes as dict
		unit = attr['Units'].decode('UTF-8').split(',')[0] # extract and decode Units
		if unit != 'degree':
			warnings.warn('Non-degree unit: ' + unit)
		return numpy.array(f.variables['SourcePosition'],dtype='float')

	@staticmethod
	def _sofa_get_listener(f):
		'''Returns dict with listener attributes from a sofa file handle.
		Keys: pos, view, up, viewvec, upvec. Used for adding a listener vector in plot functions.'''
		lis = {}
		lis['pos'] = numpy.array(f.variables['ListenerPosition'],dtype='float')[0]
		lis['view']= numpy.array(f.variables['ListenerView'],dtype='float')[0]
		lis['up']  = numpy.array(f.variables['ListenerUp'],dtype='float')[0]
		lis['viewvec'] = numpy.concatenate([lis['pos'],lis['pos']+lis['view']])
		lis['upvec'] = numpy.concatenate([lis['pos'],lis['pos']+lis['up']])
		return lis

	@staticmethod
	def _sofa_get_FIR(f):
		'Returns an array of FIR filters for all source positions from a sofa file handle.'
		datatype = f.attrs['DataType'].decode('UTF-8') # get data type
		if datatype != 'FIR':
			warnings.warn('Non-FIR data: ' + datatype)
		return numpy.array(f.variables['Data.IR'],dtype='float')

	# instance methods
	def elevations(self):
		'Return the list of sources'
		return sorted(list(set(self.sources[:,1])))

	def plot_tf(self,sourceidx,ear,linesep=20,filename=[]):
		"Plots a transfer functions of FIR filters for a given ear ['left','right'] at a given sourcepositions index"
		n = 0
		if ear == 'left':
			data = self.left.data
		elif ear == 'right':
			data = self.right.data
		else:
			raise ValueError("Unknown value for ear. Use 'left' or 'right'")
		for s in sourceidx:
			w, h = scipy.signal.freqz(data[s])
			freqs = self.samplerate*w/(2*numpy.pi)/1000 # convert rad/sample to kHz
			plt.plot(freqs,20 * numpy.log10(abs(h)) + n,label=str(self.sources[s,1])+'˚')
			n += linesep
		#plt.xscale('log')
		plt.ylabel('Amplitude [dB]')
		plt.xlabel('Frequency [kHz]')
		#plt.legend(loc='upper left')
		plt.grid()
		plt.axis('tight')
		plt.xlim(4,18)
		#layout(fig)
		if filename:
			plt.savefig(filename)
		plt.show()

	def remove_ctf(self): # UNFINISHED, UNTESTED!!!
		'''Removes the constant (non-spatial) portion of the transfer functions from an HRTF object (in place).
		Returns the constant transfer function.'''
		ctf = []
		n = len(self.fir[:, 0])
		for idx in range(n):
			_, h = scipy.signal.freqz(self.fir[idx, 0])
			ctf += numpy.log10(abs(h))/n
		for idx in range(n):
			w, h = scipy.signal.freqz(self.fir[idx, 0])
			self.fir[idx, 0] = numpy.log10(abs(h)) - ctf
		return ctf

	def median_sources(self):
		'DOC'
		idx = numpy.where(self.sources[:,0]==0)[0]
		return sorted(idx, key=lambda x: self.sources[x,1])

	def cone_sources(self,cone):
		'Return indices of sources along an off-axis sphere slice'
		cone = numpy.sin(numpy.deg2rad(cone))
		azimuth = numpy.deg2rad(self.sources[:,0])
		elevation = numpy.deg2rad(self.sources[:,1]-90)
		x = numpy.sin(elevation) * numpy.cos(azimuth)
		y = numpy.sin(elevation) * numpy.sin(azimuth)
		eles = self.elevations()
		out = []
		for ele in eles: # for each elevation, find the source closest to the target y
			subidx, = numpy.where((self.sources[:,1]==ele) & (x>=0))
			cmin = numpy.min(numpy.abs(y[subidx]-cone))
			idx, = numpy.where( (self.sources[:,1]==ele) & (numpy.abs(y-cone)==cmin) )
			out.append(idx[0])
		return sorted(out, key=lambda x: self.sources[x,1])

	def plot_sources(self,idx=False):
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
		ax.scatter(x,y,z, c = 'b', marker='.')
		ax.scatter(0,0,0, c = 'r', marker='o')
		if self.listener: # TODO: view dir is inverted!
			x_, y_, z_, u, v, w = zip(*[self.listener['viewvec'],self.listener['upvec']])
			ax.quiver(x_, y_, z_, u, v, w, length = 0.5, colors=['r','b','r','r','b','b'])
		if idx:
			ax.scatter(x[idx],y[idx],z[idx], c='r', marker='o')
		ax.set_xlabel('X [m]')
		ax.set_ylabel('Y [m]')
		ax.set_zlabel('Z [m]')
		plt.show()
