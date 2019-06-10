'''
Zhenyu's experiment:
Is motion adaptation speed-dependent?
'''

import time
import slab

def moving_cos(duration, from_dir, to_dir, width):
	'''
	Make a wide cos-shaped stimulus that moves horizontally.
	This is the base stimulus of the experiment.
	'''
	# make cos amplitude shape
	# itd/ild ramp for each sound
	# add sounds
	# amp ramp the sum
	return None

def run(subject, speed):
	'''
	Experiment procedure
	'''
	# make the stimuli
	probe_l = moving_cos(duration=1.0, from_dir=-45, to_dir=45, width=15)
	probe_r = moving_cos(duration=1.0, from_dir=-45, to_dir=45, width=15)
	speeds = [160, 180, 200] # deg/sec
	adapters = {}
	for speed in speeds:
		adapters[speed] = {}
		jwd = moving_cos(duration=1.0, from_dir=-45, to_dir=45, width=15).repeat(times=5)
		adapters[speed]['same'] = slab.Sound.sequence([jwd, probe_l])
		adapters[speed]['diff'] = slab.Sound.sequence([jwd, probe_l])
	# set up the block sequence
	blocks = slab.Trialsequence(conditions=len(speeds), n_reps=10)
	# set up the results file
	file = slab.Resultsfile(subject=subject+str(speed))
	# iterate through the blocks
	for speed in blocks:
		# present the 60 sec pre-adapter
		t_start = time.time()
		delta_t = 0
		while delta_t < 60:
			adapters[speed]['same'].play()
			delta_t = time.time() - t_start
		# iterate through the trials
		seq = slab.Trialsequence(conditions=['same','diff'], n_reps=10, kind='random_permutation')
		for condition in seq:
			# present correct adapter-probe pair
			adapters[speed][condition].play()
			# get response
			response = None
			# write to file
			file.write(f'{speed}, {condition}, {response}')
	# calculate outcome hitrates
