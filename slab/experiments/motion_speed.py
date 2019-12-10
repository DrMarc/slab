'''
Zhenyu's experiment:
Is motion adaptation speed-dependent?
Use like this:
>>> from slab.experiments import motion_speed
>>> motion_speed.experiment('subject01')
'''

import time
import functools
import os
import numpy
import scipy
import slab

# confiuration
# could go into config file and be loaded here with cfg = slab.load_config('config.txt'),
# then variables are accessible as cfg.speaker_positions etc.
slab.Signal.set_default_samplerate(44100)
_speaker_positions = numpy.arange(-90, 0.01, 4)
_results_file = None
_adapter_speed = 150
_adapter_dir = 'left'
_n_adapters_per_trial = 4
_n_reps_in_block = 10
_n_blocks_per_speed = 2
_after_stim_pause = 0.25
_speeds = [50, 100, 150, 200, 250] # deg/sec

slab.Resultsfile.results_folder = 'Results'
_stim_folder = 'Stimuli'

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
	sig.filter(f=[500,14000], kind='bp')
	sig = sig.externalize()
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
		stim = moving_gaussian(speed=_adapter_speed, SNR=100, direction=dir)
		stim.play() # present
		with slab.Key() as key: # and get response
			resp = key.getch()
		if dir == 'left': # transform response: left = key '1', right = key '2'
			resp = resp == 49
		else:
			resp = resp == 50
		responses.append(resp)
		#_results_file.write(dir + ', ' + str(resp))
		time.sleep(_after_stim_pause)
	# compute hitrate
	hitrate = sum(responses)/trials.n_trials
	_results_file.write(hitrate, tag='hitrate')
	return hitrate

def jnd(speed):
	'''
	Presents a staricase of moving_gaussian stimuli with varying SNR and returns the threshold.
	This threshold is used in the main experiment as the listener-specific SNR parameter.
	Target is the left-moving stimulus.
	'''
	# define the staircase
	stairs = slab.Staircase(start_val=20, n_reversals=8,\
			step_sizes=[8,4,2], min_val=2, max_val=30, n_up=1, n_down=4, n_trials=40)
	# loop through it
	_results_file.write('jnd:', tag='time')
	for trial in stairs:
		# make and present stimulus with SNR=trial
		stim_l = moving_gaussian(speed=speed, SNR=trial, direction='left')
		stim_r = moving_gaussian(speed=speed, SNR=trial, direction='right')
		stairs.present_afc_trial(stim_l, stim_r)
		stairs.print_trial_info()
		stairs.plot()
		time.sleep(_after_stim_pause)
	_results_file.write(stairs.threshold(), tag=speed)
	stairs.close_plot()
	return round(stairs.threshold(), ndigits=1)

def make_stimuli(subject, jnd_snr):
	def make_adaptor_probe_pair(probe_snr, probe_speed, probe_dir):
		kwargs = {'speed': _adapter_speed, 'SNR': 100, 'direction': _adapter_dir}
		make_adapter = functools.partial(moving_gaussian, **kwargs)
		adapter_list = slab.Precomputed(make_adapter, _n_adapters_per_trial)
		adapter = slab.Sound.sequence(*adapter_list) # now we have 5 adapters in a row
		probe = moving_gaussian(speed=probe_speed, SNR=probe_snr, direction=probe_dir)
		stim = slab.Sound.sequence(adapter, probe)
		return stim
	for speed, jnd in jnd_snr.items():
		print(f'Writing stimuli for speed {speed}...')
		probe_dir_same = _adapter_dir
		if probe_dir_same == 'right':
			probe_dir_diff = 'left'
		else:
			probe_dir_diff = 'right'
		adapter_probe_same = slab.Precomputed(lambda: make_adaptor_probe_pair(probe_snr=jnd, probe_speed=speed, probe_dir=probe_dir_same), n=10)
		adapter_probe_diff = slab.Precomputed(lambda: make_adaptor_probe_pair(probe_snr=jnd, probe_speed=speed, probe_dir=probe_dir_diff), n=10)
		adapter_probe_same.write(f'{_stim_folder}{os.sep}{subject}_speed_{speed}_same.zip')
		adapter_probe_diff.write(f'{_stim_folder}{os.sep}{subject}_speed_{speed}_diff.zip')
		print(f'Written stimuli for {subject}, speed {speed}.')

def block(adapter_probe_same,adapter_probe_diff):
	# iterate through the trials
	print('Running trial sequence...')
	seq = slab.Trialsequence(conditions=10, n_reps=_n_reps_in_block, kind='random_permutation')
	conditions = list()
	for speed in _speeds:
		conditions.append((speed, 'same'))
		conditions.append((speed, 'diff'))
	responses = []
	for condition_idx in seq:
		speed, dir = conditions[condition_idx]
		if dir == 'same':
			adapter_probe_same[speed].play()
			correct = 49 # set the correct response 'yes, same', key ('1')
		else:
			adapter_probe_diff[speed].play()
			correct = 50 # set the correct response 'no, different' key ('2')
		# get response
		with slab.Key() as key:
			response = key.getch()
		response = response == correct # reponse is now True or False
		responses.append(response)
		# write to file
		_results_file.write(f'{speed}, {dir}, {response}')
		time.sleep(_after_stim_pause)
	hitrate = sum(responses)/seq.n_trials
	#_results_file.write(hitrate, tag='hitrate')
	print(f'hitrate: {hitrate}')
	return hitrate

def day1(subject=None):
	global _results_file
	# set up the results file
	if not subject:
		subject = input('Enter subject code: ')
	_results_file = slab.Resultsfile(subject=subject)

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
	jnd_snr = dict()
	for speed in _speeds:
		input('Press enter to start JND estimation (10min)...')
		repeat = 'r'
		while repeat == 'r':
			jnd_snr[speed] = jnd(speed)
			print(f'jnd for {speed} deg/sec: {jnd_snr[speed]}')
			repeat = input('Press enter to continue, "r" to repeat this threshold measurement.')
	make_stimuli(subject, jnd_snr)

def day2(subject=None):
	global _results_file
	if not subject:
		subject = input('Enter subject code: ')
	_results_file = slab.Resultsfile(subject=subject)
	print('Loading stimuli...')
	adapter_probe_same = dict()
	adapter_probe_diff = dict()
	for speed in _speeds:
		adapter_probe_same[speed] = slab.Precomputed(f'{_stim_folder}{os.sep}{subject}_speed_{speed}_same.zip')
		adapter_probe_diff[speed] = slab.Precomputed(f'{_stim_folder}{os.sep}{subject}_speed_{speed}_diff.zip')
	print('Done.')
	print('Main experiment: Each trial consists of a series of 5 sounds (4 adapters)')
	print('moving in one direction, then one sound moving either in the same or the')
	print('other direction.')
	print('Is the last sound moving in the same direction? Press 1 for yes, 2 for no.')
	input('Press enter to start experiment (30min)...')
	# iterate through the blocks
	for idx in range(_n_blocks_per_speed * len(_speeds)):
		hitrate = block(adapter_probe_same,adapter_probe_diff)
		_ = input(f'Press enter to start the block {idx+1} of {_n_blocks_per_speed * len(_speeds)} (4.5min)...')
	print('Done.')


if __name__ == '__main__':
	slab.Signal.set_default_samplerate(32000)
	#experiment(subject='test')
	_results_file = slab.Resultsfile(subject='test')
	block(speed=60, jnd_snr=7)
