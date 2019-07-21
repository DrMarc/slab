'''
Zhenyu's experiment:
Is motion adaptation speed-dependent?
Use like this:
>>> from slab.experiments import motion_speed
>>> motion_speed.experiment('subject01')
'''

import time
import functools
import numpy
import scipy
import slab

# confiuration
# could go into config file and be loaded here with cfg = slab.load_config('config.txt'),
# then variables are accessible as cfg.speaker_positions etc.
_speaker_positions = numpy.arange(-90, 0.01, 4)
_results_file = None
_probe_speed = 150
_adapter_dir = 'left'
_pre_adapter_duration = 30 # seconds
_n_reps_in_block = 10 # 20 for 20 reps same and different, 40 trials a 2s, plus pre_adapter, about 2.5 min blocks
_n_blocks_per_speed = 1 # 3 for ~35min for block presentation, plus pauses etc, ~45min experiment
_after_stim_pause = 0.25

slab.Resultsfile.results_folder = 'Results'

def moving_gaussian(speed=100, width=7.5, SNR=10, direction='left'):
	'''
	Make a wide Gauss curve shaped stimulus that moves horizontally across
	a range of virtual loudspeakers. This is the base stimulus of the experiment.
	'''
	if direction == 'left':
		dir = -1
		starting_loc = _speaker_positions[-1]
	else:
		dir = 1
		starting_loc = _speaker_positions[0]
	# make a function amplitude(pos) = f(time, pos)
	def loc(time):
		return (speed * time) * dir + starting_loc
	# make times vector from speed and positions angle difference
	end_time = _speaker_positions.ptp() / speed
	time_delta = 0.01 # 10 ms
	times = numpy.arange(0, end_time + time_delta, time_delta)
	# step through time, saving speaker amplitudes for each step
	speaker_amps = numpy.zeros((len(_speaker_positions), len(times)))
	for idx, t in enumerate(times):
		speaker_amps[:,idx] = scipy.stats.norm.pdf(_speaker_positions, loc=loc(t), scale=width)
	# scale the amplitudes to max 0, min -SNR dB
	maximum = scipy.stats.norm.pdf(0, loc=0, scale=width)
	#minimum = 0
	minimum = speaker_amps.min()
	speaker_amps = numpy.interp(speaker_amps, [minimum,maximum], [-SNR,0])
	# import matplotlib.pyplot as plt
	# plt.plot(speaker_amps)
	# plt.show()
	# make a sound for each speaker, interpolating the envelopes
	speaker_signals = []
	for i, speaker_position in enumerate(_speaker_positions):
		sig = slab.Binaural.pinknoise(duration=end_time)
		sig = sig.at_azimuth(azimuth=speaker_position)
		sig = sig.envelope(envelope=speaker_amps[i,:], times=times, kind='dB')
		speaker_signals.append(sig)
	sig = speaker_signals[0]
	for speaker_signal in speaker_signals[1:]: # add sounds
		sig += speaker_signal
	sig /= len(_speaker_positions)
	sig.ramp(duration=end_time/3) # ramp the sum
	sig.level = 75 # set to 75dB
	return sig

def familiarization():
	'''
	Presents the familiarization stimuli (100% modulation depth, random direction)
	'''
	trials = slab.Trialsequence(conditions=['left','right'], n_reps=10, kind='random_permutation')
	responses = []
	_results_file.write('familiarization:', tag='time')
	for dir in trials:
		stim = moving_gaussian(speed=_probe_speed, SNR=100, direction=dir)
		stim.play() # present
		with slab.Key() as key: # and get response
			resp = key.getch()
		if dir == 'left': # transform response: left = key '1', right = key '2'
			resp = resp == 49
		else:
			resp = resp == 50
		responses.append(resp)
		_results_file.write(dir + str(resp))
		time.sleep(_after_stim_pause)
	# compute hitrate
	hitrate = sum(responses)/trials.n_trials
	_results_file.write(hitrate, tag='hitrate')
	return hitrate

def jnd():
	'''
	Presents a staricase of moving_gaussian stimuli with varying SNR and returns the threshold.
	This threshold is used in the main experiment as the listener-specific SNR parameter.
	Target is the left-moving stimulus.
	'''
	# define the staircase
	stairs = slab.Staircase(start_val=20, n_reversals=8,\
			step_sizes=[8,4,2], min_val=2, max_val=30, n_up=1, n_down=2, n_trials=40)
	# loop through it
	_results_file.write('jnd:', tag='time')
	for trial in stairs:
		# make and present stimulus with SNR=trial
		stim_l = moving_gaussian(speed=_probe_speed, SNR=trial, direction='left')
		stim_r = moving_gaussian(speed=_probe_speed, SNR=trial, direction='right')
		stairs.present_afc_trial(stim_l, stim_r)
		stairs.print_trial_info()
		stairs.plot()
		time.sleep(_after_stim_pause)
	_results_file.write(stairs.__repr__())
	return stairs.threshold()

def block(speed=150, jnd_snr=10):
	'''
	Runs one block of the main experiment with a given speed of motion.
	'''
	def make_adaptor_probe_pair(adapter_speed, probe_dir):
		kwargs = {'speed': adapter_speed, 'SNR': 100, 'direction': _adapter_dir}
		make_adapter = functools.partial(moving_gaussian, **kwargs)
		times = round((6/(90/speed))) # repetitions to make 6s adapter
		adapter_list = slab.Precomputed(make_adapter, times)
		adapter = slab.Sound.sequence(*adapter_list) # now we have 5 adapters in a row
		probe = moving_gaussian(speed=_probe_speed, SNR=jnd_snr, direction=probe_dir)
		stim = slab.Sound.sequence(adapter, probe)
		return stim
	# precompute the pre-adapter
	print('precompute the pre-adapter')
	pre_adapter = slab.Precomputed(lambda: moving_gaussian(speed=speed, SNR=100, direction=_adapter_dir), n=3)
	# precompute adapter probe pairs (left/right, 10 each)
	probe_dir_same = _adapter_dir
	probe_dir_diff = [dir for dir in ['left', 'right'] if not dir == _adapter_dir]
	print('precompute adapter_probe pairs')
	adapter_probe_same = slab.Precomputed(lambda: make_adaptor_probe_pair(adapter_speed=speed, probe_dir=probe_dir_same), n=3)
	adapter_probe_diff = slab.Precomputed(lambda: make_adaptor_probe_pair(adapter_speed=speed, probe_dir=probe_dir_diff), n=3)
	# present the 60 sec pre-adapter
	t_start = time.time()
	delta_t = 0
	print('presenting pre-adapter')
	while delta_t < _pre_adapter_duration:
		pre_adapter.play()
		delta_t = time.time() - t_start
	time.sleep(1)
	# iterate through the trials
	print('running trial sequence')
	seq = slab.Trialsequence(conditions=['same','diff'], n_reps=_n_reps_in_block, kind='random_permutation')
	responses = []
	for condition in seq:
		# present correct adapter-probe pair
		if condition == 'same':
			adapter_probe_same.play()
			correct = 49 # set the correct response 'yes, same', key ('1')
		else:
			adapter_probe_diff.play()
			correct = 50 # set the correct response 'no, different' key ('2')
		# get response
		with slab.Key() as key:
			response = key.getch()
		response = response == correct # reponse is now True or False
		responses.append(response)
		# write to file
		_results_file.write(f'{speed}, {condition}, {response}')
		time.sleep(_after_stim_pause)
	# calculate outcome hitrates INCORRECT!!! need to sort by condition!
	hitrate = sum(responses)/seq.n_trials
	_results_file.write(hitrate, tag='hitrate')
	print(f'hitrate: {hitrate}')
	return hitrate

def experiment(subject=None):
	'''
	Main experiment
	'''
	global _results_file
	# set up the results file
	if not subject:
		subject = input('Enter subject code: ')
	_results_file = slab.Resultsfile(subject=subject)
	speeds = [30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300] # deg/sec

	print('Familiarization: sounds moving left or right are presented.')
	print('The direction should be easy to hear.')
	print('Is the sound moving to the left? Press 1 for yes, 2 for no.')
	input('Press enter to start familiarization (2min)...')
	repeat = 'r'
	while repeat == 'r':
		hitrate = familiarization()
		print(f'hitrate: {hitrate}')
		repeat = input('Press enter to continue, "r" to repeat familiarization.')

	print('Motion direction threshold: Two sounds moving left or right are presented.')
	print('Is the sound moving to the left presented first or secong? Press 1 for first, 2 for second.')
	print('The direction will get more and more difficult to hear.')
	input('Press enter to start JND estimation (10min)...')
	repeat = 'r'
	while repeat == 'r':
		jnd_snr = jnd()
		print(f'jnd: {jnd_snr}')
		repeat = input('Press enter to continue, "r" to repeat threshold measurement.')

	print('Main experiment. Each block starts with a 1min sounds, continously moving to one side.')
	print('After a short pause, a series of 5 sounds is played, moving in one direction,')
	print('then one sound moving either in the same or the other direction.')
	print('Is the sound moving in the same direction? Press 1 for yes, 2 for no.')
	input('Press enter to start experiment (30min)...')
	# set up the block sequence
	blocks = slab.Trialsequence(conditions=speeds, n_reps=_n_blocks_per_speed)
	# iterate through the blocks
	for speed in blocks:
		hitrate = block(speed, jnd_snr)
		_ = input('Press enter to start the next block (2.5min)...')
	print('Done.')


if __name__ == '__main__':
	slab.Signal.set_default_samplerate(32000)
	#experiment(subject='test')
	_results_file = slab.Resultsfile(subject='test')
	block(speed=60, jnd_snr=7)
