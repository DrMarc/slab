'''
Zhenyu's experiment:
Is motion adaptation speed-dependent?
Use like this:
>>> from slab.experiments import motion_speed
>>> motion_speed.day1('subject01')
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
_n_blocks_per_speed = 2
_after_stim_pause = 0.1
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
	#sig.filter(f=[500,14000], kind='bp')
	#sig = sig.externalize()
	sig.level = 75 # set to 75dB
	return sig

def familiarization():
	'''
	Presents the familiarization stimuli (100% modulation depth, random direction)
	'''
	print('Familiarization: sounds moving left or right are presented.')
	print('The direction should be easy to hear.')
	print('Press 1 for left, 2 for right.')
	input('Press enter to start familiarization (2min)...')
	repeat = 'r'
	while repeat == 'r':
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
		print(f'hitrate: {hitrate}')
		repeat = input('Press enter to continue, "r" to repeat familiarization.')
	_results_file.write(hitrate, tag='hitrate')
	return hitrate

def jnd(speed=None, adapter_list=None):
	'''
	Presents a staricase of moving_gaussian stimuli with varying SNR and returns the threshold.
	This threshold is used in the main experiment as the listener-specific SNR parameter.
	'''
	if adapter_list:
		print(f'{len(adapter_list)} sounds moving {_adapter_dir}, followed by')
		print('one sound moving left or right is presented.')
		print('Is this last sound moving left or right?')
	else:
		print('One sound is presented in each trial.')
		print('Is this sound moving left or right?')
	print('Press 1 for left, 2 for right.')
	print('The direction will get more and more difficult to hear.')
	input('Press enter to start JND estimation...')
	repeat = 'r'
	while repeat == 'r':
		# define the staircase
		stairs = slab.Staircase(start_val=24, n_reversals=14,\
				step_sizes=[8,6,4,2,1], min_val=1, max_val=30, n_up=1, n_down=1, n_pretrials=3)
		# loop through it
		_results_file.write('jnd:', tag='time')
		for trial in stairs:
			direction = numpy.random.choice(('left', 'right'))
			stim = moving_gaussian(speed=speed, SNR=trial, direction=direction)
			if adapter_list:
				adapters = adapter_list.random_choice(n=_n_adapters_per_trial) # some variety in the adapters
				adapters.append(stim) # add stim to list of adapters
				stim = slab.Sound.sequence(*adapters) # concatenate sounds in the list
			stairs.present_tone_trial(stimulus=stim, correct_key_idx=1 if direction == 'left' else 2)
		thresh = stairs.threshold(n=10)
		tag = f"{speed} {'with_adapter' if adapter_list else 'no_adapter'}"
		print(f'jnd for {tag}: {round(thresh, ndigits=1)}')
		repeat = input('Press enter to continue, "r" to repeat this threshold measurement.')
		_results_file.write(thresh, tag=tag)
	return thresh

def make_adapters():
	'Pre-make many adapter instances to speed-up constructing the stimuli.'
	kwargs = {'speed': _adapter_speed, 'SNR': 100, 'direction': _adapter_dir}
	make_adapter = functools.partial(moving_gaussian, **kwargs)
	adapter_list = slab.Precomputed(make_adapter, _n_adapters_per_trial * 3)
	return adapter_list

def main_experiment(subject=None):
	global _results_file
	# set up the results file
	if not subject:
		subject = input('Enter subject code: ')
	_results_file = slab.Resultsfile(subject=subject)
	_ = familiarization() # run the familiarization, the hitrate is saved in the results file
	print('The main part of the experiment starts now (motion direction thresholds).')
	adapter_list = make_adapters()
	speed_seq = slab.Trialsequence(conditions=_speeds, n_reps=_n_blocks_per_speed,\
		kind='random_permutation')
	jnds = numpy.empty((len(_speeds), 3)) # results table with three colums: speed, jnd_no_adapter, jnd_adapter
	for speed in speed_seq:
		idx = _speeds.index(speed) # index of the current speed value, used for results table
		jnds[idx,0] = speed
		if numpy.random.choice((True, False)): # presented without adapters first
			jnds[idx,1] = jnd(speed) # each call to jnd prints instructions and saves to the results file
			jnds[idx,2] = jnd(speed, adapter_list)
		else: # with adapters first
			jnds[idx,2] = jnd(speed, adapter_list)
			jnds[idx,1] = jnd(speed)
	_results_file.write(str(jnds), tag='final results') # save a string representation of the numpy results table


if __name__ == '__main__':
	main_experiment(subject='test')
