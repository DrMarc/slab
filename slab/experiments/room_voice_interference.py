'''
Michaela's experiment:
Interference between room and voice processing
Use like this:
>>> from slab.experiments import room_voice_interference
>>> room_voice_interference.main_experiment('subject01')
Please see docstring of main_experiment() for details.
'''

import time
import pathlib
import collections
import numpy
import slab

# configuration
slab.Signal.set_default_samplerate(44100)
_results_file = None
slab.ResultsFile.results_folder = 'Results'
slab.psychoacoustics.input_method = 'keyboard' # or 'buttonbox'
# possible parameters:
rooms = tuple(range(40, 161, 8)) # (40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160)
voices = (0.98, 1.029, 1.078, 1.127, 1.176, 1.225, 1.274, 1.323, 1.372, 1.421, 1.47)
itds = tuple(range(0, 401, 40)) # (0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400)
default_room = rooms[0] # 40
default_voice = voices[0] # 0.98
default_itd = itds[0] # 0
ISI_stairs = 0.15
_after_stim_pause = 0.3
word_list = ['Aertel', 'Apor', 'Aucke', 'Bohke', 'Dercke', 'Eika', 'Eukof', 'Felcke', 'Geke', 'Gelkat', 'Kansup', 'Kelpeit', 'Kirpe', 'Kitlu', 'Klamsup', 'Kontus', 'Lanrapf', 'Manzeb', 'Nukal', 'Pekus', 'Perkus', 'Raupfan', 'Reiwat', 'Repad', 'Retel', 'Schaujak', 'Seckuck', 'Sekaak', 'Stiecke', 'Subter', 'Trepfel', 'Tunsat', 'Verzung', 'Waatep', 'Wieken', 'Zeten']
stim_folder = pathlib.Path('..') / pathlib.Path('Stimuli') # set to correct path
condition = collections.namedtuple('condition', ['voice', 'room', 'itd', 'label']) # used to set parameters in interference_block

def jnd(condition, practise=False):
    '''
    Presents a staricase for a 2AFC task and returns the threshold.
    This threshold is used in the main experiment as jnd.
    condition ... 'room', voice', or 'itd'
    '''
    print('Three sounds are presented in each trial.')
    print('They are always different, but sometimes')
    if condition == 'room':
        print('one sound is played in a larger room,')
        print('and sometimes all three are played in the same room.')
        print('Was the larger room presented first, second, or third?')
    elif condition == 'voice':
        print('one is spoken by a different (larger) person,')
        print('and sometimes all three are spoken by the same person.')
        print('Was the larger person presented first, second, or third?')
    elif condition == 'itd':
        print('one is played from a different direction (slightly to the left),')
        print('and sometimes all three are played from straight ahead.')
        print('Was the sound slightly from the left played first, second, or third?')
    else:
        raise ValueError(f'Invalid condition {condition}.')
    print('Press 1 for first, 2 for second, 3 for third.')
    print('The difference will get more and more difficult to hear.')
    input('Press enter to start JND estimation...')
    repeat = 'r'
    condition_values = globals()[condition+'s'] # get the parameter list (vars rooms, voices, or itds) from condition string
    while repeat == 'r':
        # make a random, non-repeating list of words to present during the staircase
        word_seq = slab.Trialsequence(conditions=word_list, kind='infinite', label='word_seq')
        # define the staircase
        if practise:
            stairs = slab.Staircase(start_val=len(condition_values)-1, n_reversals=3,
                                step_sizes=[4, 3, 2], min_val=0, max_val=len(condition_values)-1, n_up=1, n_down=1, n_pretrials=0)
        else:
            stairs = slab.Staircase(start_val=len(condition_values)-4, n_reversals=15,
                                step_sizes=[4, 2], min_val=0, max_val=len(condition_values)-1, n_up=1, n_down=2, step_up_factor=1.5, n_pretrials=1) # should give approx. 70% hitrate
            _results_file.write(f'{condition} jnd:', tag='time')
        for trial in stairs:
            current = condition_values[int(trial)]
            # load stimuli
            word = next(word_seq)
            word2 = next(word_seq)
            word3 = next(word_seq)
            if condition == 'room':
                jnd_stim = slab.Sound(stim_folder / word  / f'{word}_SER{default_voice:.4g}_GPR168_{current}_{default_itd}.wav')
            elif condition == 'voice':
                jnd_stim = slab.Sound(stim_folder / word  / f'{word}_SER{current:.4g}_GPR168_{default_room}_{default_itd}.wav')
            elif condition == 'itd':
                jnd_stim = slab.Sound(stim_folder / word  / f'{word}_SER{default_voice:.4g}_GPR168_{default_room}_{current}.wav')
            default_stim1 = slab.Sound(stim_folder / word2 / f'{word2}_SER{default_voice:.4g}_GPR168_{default_room}_{default_itd}.wav')
            default_stim2 = slab.Sound(stim_folder / word3 / f'{word3}_SER{default_voice:.4g}_GPR168_{default_room}_{default_itd}.wav')
            stairs.present_afc_trial(jnd_stim, [default_stim1, default_stim2], isi=ISI_stairs, print_info=practise)
            if practise:
                stairs.plot()
        thresh = stairs.threshold()
        thresh_condition_value = condition_values[numpy.ceil(thresh).astype('int')]
        if practise:
            stairs.close_plot()
        else:
            print(f'room jnd: {round(thresh, ndigits=1)}')
            _results_file.write(repr(stairs), tag=f'stairs {condition}')
            _results_file.write(thresh, tag=f'jnd {condition}')
            _results_file.write(thresh_condition_value, tag=f'jnd condition value {condition}')
        repeat = input('Press enter to continue, "r" to repeat this threshold measurement.\n\n')
    return thresh_condition_value

# same-diff task, method of constant stimuli, 5 conditions, 40 reps (20 diff, 20 same)
def interference_block(jnd_room, jnd_voice, jnd_itd):
    '''
    Presents one condition block of the the interference test.
    Condition ... 'room', 'room+voice', 'room+itd', 'voice', or 'itd'
    default_room etc. ... the reference room, SER and ITD values.
    jnd_room etc. ... the room, SER and ITD values that are perceived as different from the default
                      (default value + measured jnd rounded to the nearest available stimulus)
    '''
    print('Three sounds are presented in each trial.')
    print('They are always different, but sometimes')
    print('one sound is played in a larger room,')
    print('and sometimes all three are played in the same room.')
    print('Was the larger room presented first, second, or third?')
    print('Press 1 for first, 2 for second, and 3 for third.')
    input('Press enter to start the test...')
    # set parameter values of conditions in named tuples -> list of these is used for slab.Trialsequence
    default = condition(voice=default_voice, room=default_room, itd=default_itd, label='default')
    room = condition(voice=default_voice, room=jnd_room, itd=default_itd, label='room')
    room_voice = condition(voice=jnd_voice, room=jnd_room, itd=default_itd, label='room_voice')
    room_itd = condition(voice=default_voice, room=jnd_room, itd=jnd_itd, label='room_itd')
    voice = condition(voice=jnd_voice, room=default_room, itd=default_itd, label='voice')
    itd = condition(voice=default_voice, room=default_room, itd=jnd_itd, label='itd')
    conditions = [default, room, room_voice, room_itd, voice, itd]
    trials = slab.Trialsequence(conditions=conditions, n_reps=10, kind='random_permutation')
    word_seq = slab.Trialsequence(conditions=word_list, kind='infinite', label='word_seq')
    hits = 0
    false_alarms = 0
    _results_file.write(f'interference block:', tag='time')
    for trial_parameters in trials:
        # load stimuli
        word  = next(word_seq)
        word2 = next(word_seq)
        word3 = next(word_seq)
        jnd_stim = slab.Sound(str(stim_folder / word  / word) + '_SER%.4g_GPR168_%i_%i.wav' % trial_parameters[:-1])
        default_stim1 = slab.Sound(str(stim_folder / word2 / word2) + '_SER%.4g_GPR168_%i_%i.wav' % default[:-1])
        default_stim2 = slab.Sound(str(stim_folder / word3 / word3) + '_SER%.4g_GPR168_%i_%i.wav' % default[:-1])
        trials.present_afc_trial(jnd_stim, [default_stim1, default_stim2], isi=ISI_stairs)
        response = trials.data[-1] # read out the last response
        if trial_parameters.label[:4] == 'room' and response: # hit!
            hits += 1
        elif trial_parameters.label[:3] in ['voi', 'itd'] and response: # false alarm!
            false_alarms += 1
        time.sleep(_after_stim_pause)
    hitrate = hits/trials.n_trials
    print(f'hitrate: {hitrate}')
    farate = false_alarms/trials.n_trials
    print(f'false alarm rate: {farate}')
    _results_file.write(repr(trials), tag='trials')

def main_experiment(subject=None, do_jnd=True, do_interference=True):
    '''
    Interference between room and voice processing.
    Pre-recorded voice recordings are presented in different simulated rooms (the large stimulus
    set is not included). Just-noticeable differences for changes in room volume and voice
    parameters (glottal pulse rate and vocal tract length) are first measured, then 3-alternative-
    forced-choice trial are presented with the reference in a larger room. Does a simultaneous voice
    change impede the detection of the room change?
    The experiment requires a set of recorded spoken word stimuli, each of which was offline manipulated
    to change speaker identity (using the STRAIGHT algorithm) and then run through a room acoustics
    simulation to add reverberation consistent with rooms of different sizes. The filenames of the recordings
    contain the word and the voice and room parameters, so that the correct file is loaded for presentation.

    This experiment showcases participant data handling, AFC trials, prerecorded stimulus handling, among others.
    '''
    global _results_file
    # set up the results file
    if not subject:
        subject = input('Enter subject code: ')
    _results_file = slab.ResultsFile(subject=subject)
    if do_jnd:
        jnd('room', practise=True)  # run the stairs practice for the room condition
        jnd_room = jnd('room') # mesure
        jnd('voice', practise=True)  # run the stairs practice for the room condition
        jnd_voice = jnd('voice')
        jnd('itd', practise=True)  # run the stairs practice for the room condition
        jnd_itd = jnd('itd')
    else: # need to get the jnds from the resultsfile
        prev = slab.ResultsFile.previous_file(subject=subject)
        jnd_room = slab.ResultsFile.read_file(prev, tag='jnd_room')
        if not jnd_room:
            raise ValueError('jnd_room not found in previous results file. Please run the full experiment.')
        jnd_voice = slab.ResultsFile.read_file(prev, tag='jnd_voice')
        if not jnd_voice:
            raise ValueError('jnd_voice not found in previous results file. Please run the full experiment.')
        jnd_itd = slab.ResultsFile.read_file(prev, tag='jnd_itd')
        if not jnd_itd:
            raise ValueError('jnd_itd not found in previous results file. Please run the full experiment.')
    if do_interference:
        print('The main part of the experiment starts now (interference task).')
        print('Blocks of about 4min each are presented with pauses inbetween.')
        for _ in range(10):
            interference_block(jnd_room, jnd_voice, jnd_itd)

if __name__ == '__main__':
    main_experiment()
