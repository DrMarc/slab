'''
psychoacoustics exports classes for handling psychophysical procedures and
measures, like trial sequences and staircases.
This module uses doctests. Use like so:
python -m doctest psychoacoustics.py
'''
import os
from pathlib import Path
import datetime
import json
import zipfile
from contextlib import contextmanager
try:
    import curses
    have_curses = True
except ImportError:
    have_curses = False
import collections
import numpy
try:
    import matplotlib.pyplot as plt
    have_pyplot = True
except ImportError:
    have_pyplot = False
import slab

results_folder = 'Results'
input_method = 'keyboard' # or 'buttonbox'

class buttonbox:
    '''Class to allow easy switching between input from the keyboard via curses
    and from the custon buttonbox adapter (arduino device that sends a number keystroke
    followed by a return keystroke when pressing a button on the arduino).'''
    @staticmethod
    def getch():
        return int(input()) # buttonbox adapter has to return the keycode of intended keys!

@contextmanager
def Key():
    '''
    Wrapper for curses module to simplify getting a single keypress from the terminal.
    Use like this:
    with slab.Key() as key:
            response = key.getch()
    '''
    if input_method == 'keyboard':
        if not have_curses:
            raise ImportError(
                'You need curses to use the keypress class (pip install curses (or windows-curses))')
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        yield stdscr
        curses.nocbreak()
        curses.echo()
        curses.endwin()
    else:
        yield buttonbox


class LoadSaveJson_mixin:
    'Mixin to provide JSON loading and saving functions'

    def save_json(self, file_name=None):
        """
        Serialize the object to the JSON format.
        fileName: string, or None
                the name of the file to create or append. If `None`,
                will not write to a file, but return an in-memory JSON object.
        """
        # self_copy = copy.deepcopy(self) use if reading the json file sometimes fails
        def default(o): return int(o) if isinstance(o, numpy.int64) else o
        if (file_name is None) or (file_name == 'stdout'):
            return json.dumps(self.__dict__, indent=2, default=default)
        else:
            try:
                with open(file_name, 'w') as f:
                    json.dump(self.__dict__, f, indent=2, default=default)
                    return True
            except OSError:
                return False

    def load_json(self, file_name):
        """
        Read JSON file and deserialize the object into self.__dict__.
        file_name: string, the name of the file to read.
        """
        with open(file_name, 'r') as f:
            self.__dict__ = json.load(f)


class TrialPresentationOptions_mixin:
    '''Mixin to provide AFC and Same-Different trial presentation methods and
    response simulation to Trialsequence and Staircase.'''

    def present_afc_trial(self, target, distractors, key_codes=(range(49, 58)), isi=0.25, print_info=True):
        '''
        Present the target (slab sound object) in random order together
        with the distractor sound object (or list of several sounds) with
        isi pause (in seconds) in between, then aquire a response keypress
        via Key(), compare the response to the target interval and record
        the response via add_response. If key_codes for buttons are given
        (get with: ord('1') for instance -> ascii code of key 1 is 49),
        then these keys will be used as answer keys. Default are codes for
        buttons '1' to '9'.
        This is a convenience function for implementing alternative forced
        choice trials. In each trial, generate the target stimulus and
        distractors, then call present_afc_trial to play them and record
        the response. Optionally call print_trial_info afterwards.
        '''
        if isinstance(distractors, list):
            stims = [target].extend(distractors)  # assuming sound object and list of sounds
        else:
            stims = [target, distractors]  # assuming two sound objects
        order = numpy.random.permutation(len(stims))
        print(order)
        for idx in order:
            stim = stims[idx]
            stim.play()
            plt.pause(isi)
        with Key() as key:
            response = key.getch()
        interval = numpy.where(order == 0)[0][0]
        print(interval)
        interval_key = key_codes[interval]
        print(interval_key)
        print(response)
        response = response == interval_key
        self.add_response(response)
        if print_info:
            self.print_trial_info()

    def present_tone_trial(self, stimulus, correct_key_idx, key_codes=(range(49, 58)), print_info=True):
        stimulus.play()
        with slab.Key() as key:
            response = key.getch()
        response = response == key_codes[correct_key_idx]
        self.add_response(response)
        if print_info:
            self.print_trial_info()

    def simulate_response(self, thresh, transition_width=2, intervals=1, hitrates=None):
        '''Return a simulated response to the current condition index value by calculating the hitrate
        from a psychometric (logistic) function. This is only sensible for 'method of constant stimuli'
        trials (self.trials has to be numeric and an interval scale representing a continuous stimulus value.
        thresh ... midpoint/threshhold (*2*)
        transition_width ... range of stimulus intensities over which the hitrate increases from 0.25 to 0.75
        intervals ... use 1 (default) to indicate a yes/no trial, 2 or more to indicate an AFC trial.
        hitrates ... list of hitrates for the different conditions, to allow costum rates instead of simulation.
                     If given, thresh and transition_width are not used.
        '''
        slope = 0.5 / transition_width
        if self.__class__.__name__ == 'Trialsequence':
            current_condition = self.trials[self.this_n]
        else:
            current_condition = self._next_intensity
        if hitrates is None:
            hitrate = 1 / (1 + numpy.exp(4 * slope  * (thresh - current_condition))) # scale/4  = slope at midpoint
        else:
            hitrate = hitrates[current_condition]
        hit = numpy.random.rand() < hitrate # True with probability hitrate
        if hit or intervals == 1:
            return hit
        else: # stim/difference not detected and AFC trial
            return numpy.random.rand() < 1/intervals # still 1/intervals chance to hit the right interval


class Trialsequence(collections.abc.Iterator, LoadSaveJson_mixin, TrialPresentationOptions_mixin):  # TODO: correct string conditions!
    """Non-adaptive trial sequences
    Parameters:
    conditions: an integer, list, or flat array specifying condition indices,
            or a list of strings or other objects (dictionaries/tuples/namedtuples)
            specifying names or stimulus values for each condition.
            If given an integer x, uses range(x).
            If conditions is a string, then it is treated as the name of a previously
            saved trial sequence object, which is then loaded.
    n_reps: number of repeats of each condition (total trial number = len(conditions) * n_reps)
    trials: a list of conditions, i.e. the trial sequence. Typically, this
            list is left empty and generated by the class based on the other parameters.
    kind: The kind of sequence randomization used to generate the trial sequence.
            'non_repeating' (conditions are repeated in randome order n_reps times without
            direct repetition - default if n_conds > 2),
            'random_permutation' (conditions are permuted randomly without control over
            transition probabilities - default if n_conds <= 2), or
            'infinite' (non_repeating [if n_conds <= 2] or random_permutation trial sequence,
            reset when end is reached to generate an infinite number of trials).
    name: a text label for the sequence.

    Attributes:
    .n_trials - the total number of trials that will be run
    .n_remaining - the total number of trials remaining
    .this_n - trial index in entire sequence, equals total trials completed so far
    .this_rep_n - index of repetition of the conditions we are currently in
    .this_trial_n - trial index within this repetition
    .this_trial - a dictionary giving the parameters of the current trial
    .finished - True/False: have we finished yet?
    .kind - records the kind of sequence ('random_permutation', 'non_repeating', 'infinite')
"""
# TODO: implementation if infinite sequence is a bit of a hack (number of completed trials needs
# to be calculated as: trials.this_rep_n * trials.n_conds + trials.this_trial_n + 1)
# TODO: ability to record trials like in staircase needed?
    def __init__(self, conditions=2, n_reps=1, trials=[], kind=None, name=''):
        self.name = name
        self.n_reps = int(n_reps)
        self.conditions = conditions
        if isinstance(conditions, str) and os.path.isfile(conditions):
            self.load_json(conditions)  # import entire object from file
        elif isinstance(conditions, int):
            self.conditions = list(range(conditions))
        else:
            self.conditions = conditions
        self.n_conds = len(self.conditions)
        self.this_rep_n = 0  # index of repetition of the conditions we are currently in
        self.this_trial_n = -1  # trial index within this repetition
        self.this_n = -1 # trial index in entire sequence
        self.this_trial = [] # condition of current trial
        self.finished = False
        self.data = [] # holds responses if TrialPresentationOptions methods are called
        # generate stimulus sequence
        if not trials:
            if not kind:
                kind = 'random_permutation' if self.n_conds <= 2 else 'non_repeating'
            if kind == 'non_repeating':
                self.trials = Trialsequence._create_simple_sequence(len(self.conditions), self.n_reps)
            elif kind == 'random_permutation':
                self.trials = Trialsequence._create_random_permutation(len(self.conditions), self.n_reps)
            elif kind == 'infinite':
                if self.n_conds <= 2:
                    self.trials = Trialsequence._create_random_permutation(len(self.conditions), 5)
                else:
                    self.trials = Trialsequence._create_simple_sequence(len(self.conditions), 1)
            else:
                raise ValueError(f'Unknown kind parameter: {kind}!')
        self.n_trials = len(self.trials)
        self.n_remaining = self.n_trials
        self.kind = kind

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return f'Trialsequence, trials {"inf" if self.kind=="infinite" else self.n_trials}, remaining {"inf" if self.kind=="infinite" else self.n_remaining}, current condition {self.this_trial}'

    def __next__(self):
        """Advances to next trial and returns it.
        Updates attributes; this_trial, this_trial_n
        If the trials have ended this method will raise a StopIteration error.
        trials = Trialsequence(.......)
                for eachTrial in trials:  # automatically stops when done
        """
        self.this_trial_n += 1
        self.this_n += 1
        self.n_remaining -= 1
        if self.this_trial_n >= self.n_conds: # start a new repetition
            self.this_trial_n = 0
            self.this_rep_n += 1
        if self.n_remaining < 0:  # all trials complete
            if self.kind == 'infinite': # finite sequence -> reset and start again
                self.trials = Trialsequence._create_simple_sequence(len(self.conditions), 1,
                                previous=self.trials[-1]) # new sequence, avoid start with previous condition
                self.this_n = 0
                self.n_remaining = self.n_trials - 1 # reset trial countdown to length of new trial sequence
                                # (subtract 1 because we return the 0th trial below)
            else: # finite sequence -> finish
                self.this_trial = []
                self.finished = True
        if self.finished:
            raise StopIteration
        self.this_trial = self.conditions[self.trials[self.this_n]]  # fetch the trial info
        return self.this_trial

    def add_response(self, response):
        self.data.append(response)

    def print_trial_info(self):
        print(f'trial # {self.this_n} of {"inf" if self.kind=="infinite" else self.n_trials} ({"inf" if self.kind=="infinite" else self.n_remaining} remaining): condition {self.this_trial}, last response: {self.data[-1] if self.data else None}')

    @staticmethod
    def _create_simple_sequence(n_conditions, n_reps, previous=1):
        '''Create a sequence of n_conditions x n_reps trials, where each repetitions
        contains all conditions in random order, and no condition is directly
        repeated across repetitions.
        'previous' can be set to an index in range(n_conditions), which ensures that
        the sequence does not start with this index.'''
        permute = list(range(n_conditions))
        trials = [previous]
        for rep in range(n_reps):
            numpy.random.shuffle(permute)
            while trials[-1] == permute[0]:
                numpy.random.shuffle(permute)
            trials += permute
        trials = trials[1:] # delete first entry ('previous')
        return trials

    @staticmethod
    def _create_random_permutation(n_conditions, n_reps):
        '''Create a sequence of n_conditions x n_reps trials in random order.'''
        return list(numpy.random.permutation(numpy.tile(list(range(n_conditions)), n_reps)))

    def get_future_trial(self, n=1):
        """Returns the condition for n trials into the future or past,
        without advancing the trials. A negative n returns a previous (past)
        trial. Returns 'None' if attempting to go beyond the last trial.
        """
        if n > self.n_remaining or self.this_n + n < 0:
            return None
        return self.conditions[self.trials[self.this_n + n]]

    def transitions(self):
        'Return array (n_conds x n_conds) of transition probabilities.'
        transitions = numpy.zeros((self.n_conds, self.n_conds))
        for i, j in zip(self.trials, self.trials[1:]):
            transitions[i, j] += 1
        return transitions

    def condition_probabilities(self):
        'Return list of frequencies of conditions in the order listed in .conditions'
        probs = []
        for i in range(self.n_conds):
            num = self.trials.count(i)
            num /= self.n_trials
            probs.append(num)
        return probs

    def plot(self):
        'Plot the trial sequence as scatter plot.'
        if not have_pyplot:
            raise ImportError('Plotting requires matplotlib!')
        plt.plot(self.trials)
        plt.xlabel('Trials')
        plt.ylabel('Condition index')
        plt.show()

    @staticmethod
    def mmn_sequence(n_trials, deviant_freq=0.12):
    # TODO: integrate in main constructor
        '''Returns a  MMN experiment: 2 different stimuli (conditions),
        between two deviants at least 3 standards
        n_trials: number of trials to return
        deviant_freq: frequency of deviants (*0.12*, max. 0.25)
        '''
        n_partials = int(numpy.ceil((2 / deviant_freq) - 7))
        reps = int(numpy.ceil(n_trials/n_partials))
        partials = []
        for i in range(n_partials):
            partials.append([0] * (3+i) + [1])
        idx = list(range(n_partials)) * reps
        numpy.random.shuffle(idx)  # randomize order
        trials = []  # make the trial sequence by putting possibilities together
        for i in idx:
            trials.extend(partials[i])
        trials = trials[:n_trials]  # cut the list to the requested numner of trials
        return Trialsequence(conditions=2, n_reps=1, trials=trials)


class Staircase(collections.abc.Iterator, LoadSaveJson_mixin, TrialPresentationOptions_mixin):
    # TODO: add QUEST or Bayesian estimation?
    """Class to handle smoothly the selection of the next trial
    and report current values etc.
    Calls to next() will fetch the next object given to this
    handler, according to the method specified.
    The staircase will terminate when *n_trials* AND *n_reversals* have
    been exceeded. If *step_sizes* was an array and has been exceeded
    before n_trials is exceeded then the staircase will continue.
    n_up and n_down are always considered as 1 until the first reversal
    is reached. The values entered as arguments are then used.
    Lewitt (1971) gives the up-down values for different threshold points
    on the psychometric function: 1-1 (0.5), 1-2 (0.707), 1-3 (0.794),
    1-4 (0.841), 1-5 (0.891).
    Example:
    >>> stairs = Staircase(start_val=50, n_reversals=10, step_type='lin',\
                    step_sizes=[4,2], min_val=10, max_val=60, n_up=1, n_down=1, n_trials=10)
    >>> print(stairs)
    <class 'psychoacoustics.Staircase'> 1up1down, trial -1, 0 reversals of 10
    >>> for trial in stairs:
    ... 	response = stairs.simulate_response(30)
    ... 	stairs.add_response(response)
    >>> print(f'reversals: {stairs.reversal_intensities}')
    reversals: [26, 30, 28, 30, 28, 30, 28, 30, 28, 30]
    >>> print(f'mean of final 6 reversals: {stairs.threshold()}')
    mean of final 6 reversals: 28.982753492378876
    """

    def __init__(self, start_val, n_reversals=None, step_sizes=4, step_up_factor=1, n_pretrials=4, n_up=1,
                 n_down=2, step_type='lin', min_val=None, max_val=None, name=''):
        """
        :Parameters:
                name:
                        A text label.
                start_val:
                        The initial value for the staircase.
                n_reversals:
                        The minimum number of reversals permitted.
                        If `step_sizes` is a list, but the minimum number of
                        reversals to perform, `n_reversals`, is less than the
                        length of this list, PsychoPy will automatically increase
                        the minimum number of reversals and emit a warning.
                step_sizes:
                        The size of steps as a single value or a list (or array).
                        For a single value the step size is fixed. For an array or
                        list the step size will progress to the next entry
                        at each reversal.
                step_up_factor:
                        Allows different sizes for up and down steps to implement
                        a Kaernbach1991 weighted up-down method. step_sizes sets down steps,
                        which are multiplied by step_up_factor to obtain up step sizes.
                        The default is 1, i.e. same size for up and down steps.
                n_pretrials:
                        The number of pretrials presented as familiarization before
                        the actual experiment. start_val is used presentation level.
                n_up:
                        The number of 'incorrect' (or 0) responses before the
                        staircase level increases.
                n_down:
                        The number of 'correct' (or 1) responses before the
                        staircase level decreases.
                step_type: *'lin'*, 'db', 'log'
                        The type of steps that should be taken each time. 'lin'
                        will simply add or subtract that amount each step, 'db'
                        and 'log' will step by a certain number of decibels or
                        log units (note that this will prevent your value ever
                        reaching zero or less)
                min_val: *None*, or a number
                        The smallest legal value for the staircase, which can be
                        used to prevent it reaching impossible contrast values,
                        for instance.
                max_val: *None*, or a number
                        The largest legal value for the staircase, which can be
                        used to prevent it reaching impossible contrast values,
                        for instance.
        """
        self.name = name
        self.start_val = start_val
        self.n_up = n_up
        self.n_down = n_down
        self.step_type = step_type
        try:
            self.step_sizes = list(step_sizes)
        except TypeError:
            self.step_sizes = [step_sizes]
        self._variable_step = True if len(self.step_sizes) > 1 else False
        self.step_up_factor = step_up_factor
        self.step_size_current = self.step_sizes[0]
        if n_reversals is None:
            self.n_reversals = len(self.step_sizes)
        elif len(self.step_sizes) > n_reversals:
            print(
                f'Increasing number of minimum required reversals to the number of step sizes, {len(self.step_sizes)}')
            self.n_reversals = len(self.step_sizes)
        else:
            self.n_reversals = n_reversals
        self.finished = False
        self.n_pretrials = n_pretrials
        self.this_trial_n = -n_pretrials
        self.data = []
        self.intensities = []
        self.reversal_points = []
        self.reversal_intensities = []
        self.current_direction = 'down'
        self.correct_counter = 0
        self._next_intensity = self.start_val
        self.min_val = min_val
        self.max_val = max_val
        self.pf_intensities = None  # psychometric function, auto set when finished
        self.pf_percent_correct = None  # psychometric function, auto set when finished
        self.pf_responses_per_intensity = None  # psychometric function, auto set when finished

    def __next__(self):
        """Advances to next trial and returns it.
        Updates attributes; `this_trial`, `this_trial_n` and `thisIndex`.
        If the trials have ended, calling this method will raise a
        StopIteration error. This can be handled with code such as::
                staircase = Staircase(.......)
                for eachTrial in staircase:  # automatically stops when done
                        # do stuff
        """
        if not self.finished:
            self.this_trial_n += 1  # update pointer for next trial
            self.intensities.append(self._next_intensity)
            return self._next_intensity
        else:
            self._psychometric_function()  # tally responses to create a psychometric function
            raise StopIteration

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return f'Staircase {self.n_up}up-{self.n_down}down, trial {self.this_trial_n}, {len(self.reversal_intensities)} reversals of {self.n_reversals}'

    def add_response(self, result, intensity=None):
        """Add a True or 1 to indicate a correct/detected trial
        or False or 0 to indicate an incorrect/missed trial.
        This is essential to advance the staircase to a new intensity level.
        Supplying an `intensity` value indicates that you did not use
        the recommended intensity in your last trial and the staircase will
        replace its recorded value with the one supplied.
        """
        if self._next_intensity <= self.min_val:  # always record False if at min_val
            result = False
        else:
            result = bool(result)
        self.data.append(result)
        if intensity is not None:
            self.intensities.pop()
            self.intensities.append(intensity)
        if self.this_trial_n > 0:  # we're out of the pretrials
            if result:  # correct response
                if len(self.data) > 1 and self.data[-2] == result:
                    self.correct_counter += 1  # increment if on a run
                else:
                    self.correct_counter = 1  # or reset
            else:  # incorrect response
                if len(self.data) > 1 and self.data[-2] == result:
                    self.correct_counter -= 1  # decrement if on a run
                else:
                    self.correct_counter = -1  # or reset
            self.calculatenext_intensity()

    def calculatenext_intensity(self):
        'Based on current intensity, counter of correct responses, and current direction.'
        if not self.reversal_intensities:  # no reversals yet
            if self.data[-1] is True:  # last answer correct
                reversal = bool(self.current_direction == 'up')  # got it right
                self.current_direction = 'down'
            else:  # got it wrong
                reversal = bool(self.current_direction == 'down')
                self.current_direction = 'up'
        elif self.correct_counter >= self.n_down:  # n right, time to go down!
            reversal = bool(self.current_direction != 'down')
            self.current_direction = 'down'
        elif self.correct_counter <= -self.n_up:  # n wrong, time to go up!
            reversal = bool(self.current_direction != 'up')
            self.current_direction = 'up'
        else:  # same as previous trial
            reversal = False
        if reversal:  # add reversal info
            self.reversal_points.append(self.this_trial_n)
            self.reversal_intensities.append(self.intensities[-1])
        if len(self.reversal_intensities) >= self.n_reversals:
            self.finished = True  # we're done
        if reversal and self._variable_step:  # new step size if necessary
            # if beyond the list of step sizes, use the last one
            if len(self.reversal_intensities) >= len(self.step_sizes):
                self.step_size_current = self.step_sizes[-1]
            else:
                _sz = len(self.reversal_intensities)
                self.step_size_current = self.step_sizes[_sz]
        if self.current_direction == 'up':
            self.step_size_current *= self.step_up_factor # apply factor for weighted up/down method
        if not self.reversal_intensities:
            if self.data[-1] == 1:
                self._intensity_dec()
            else:
                self._intensity_inc()
        elif self.correct_counter >= self.n_down:
            self._intensity_dec()  # n right, so going down
        elif self.correct_counter <= -self.n_up:
            self._intensity_inc()  # n wrong, so going up

    def _intensity_inc(self):
        'increment the current intensity and reset counter'
        if self.step_type == 'db':
            self._next_intensity *= 10.0**(self.step_size_current/20.0)
        elif self.step_type == 'log':
            self._next_intensity *= 10.0**self.step_size_current
        elif self.step_type == 'lin':
            self._next_intensity += self.step_size_current
        if (self.max_val is not None) and (self._next_intensity > self.max_val):
            self._next_intensity = self.max_val  # check we haven't gone out of the legal range
        self.correct_counter = 0

    def _intensity_dec(self):
        'decrement the current intensity and reset counter'
        if self.step_type == 'db':
            self._next_intensity /= 10.0**(self.step_size_current/20.0)
        if self.step_type == 'log':
            self._next_intensity /= 10.0**self.step_size_current
        elif self.step_type == 'lin':
            self._next_intensity -= self.step_size_current
        self.correct_counter = 0
        if (self.min_val is not None) and (self._next_intensity < self.min_val):
            self._next_intensity = self.min_val  # check we haven't gone out of the legal range

    def threshold(self, n=6):
        '''Returns the average (arithmetic for step_type == 'lin',
        geometric otherwise) of the last n reversals (default 6).'''
        if self.finished:
            if n > self.n_reversals:
                n = self.n_reversals
            if self.step_type == 'lin':
                return numpy.mean(self.reversal_intensities[-n:])
            else:
                return numpy.exp(numpy.mean(numpy.log(self.reversal_intensities[-n:])))

    def print_trial_info(self):
        print(
            f'trial # {self.this_trial_n}: reversals: {len(self.reversal_points)}/{self.n_reversals}, intensity {round(self.intensities[-1],2) if self.intensities else round(self._next_intensity,2)}, going {self.current_direction}, response {self.data[-1] if self.data else None}')

    def save_csv(self, fileName):
        'Write a text file with the data.'
        if self.this_trial_n < 1:
            return -1  # no trials to save
        with open(fileName, 'w') as f:
            raw_intens = str(self.intensities)
            raw_intens = raw_intens.replace('[', '').replace(']', '')
            f.write(raw_intens)
            f.write('\n')
            responses = str(numpy.multiply(self.data, 1))  # convert to 0 / 1
            responses = responses.replace('[', '').replace(']', '')
            responses = responses.replace(' ', ', ')
            f.write(responses)

    def plot(self):
        'Plot the staircase. If called after each trial, one plot is created and updated.'
        if not have_pyplot:
            raise ImportError('Plotting requires matplotlib!')
        x = numpy.arange(-self.n_pretrials, len(self.intensities)-self.n_pretrials)
        y = numpy.array(self.intensities)
        responses = numpy.array(self.data)
        fig = plt.figure('stairs')  # figure 'stairs' is created or made current
        plt.clf()
        plt.plot(x, y)
        ax = plt.gca()
        ax.set_xlim(-self.n_pretrials, min(20, (self.this_trial_n + 15)//10*10))  # plot
        ax.set_ylim(self.min_val, self.max_val)
        # plot green dots at correct/yes responses
        ax.scatter(x[responses], y[responses], color='green')
        # plot red dots at correct/yes responses
        ax.scatter(x[~responses], y[~responses], color='red')
        ax.set_ylabel('Dependent variable')
        ax.set_xlabel('Trial')
        ax.set_title('Staircase')
        if self.finished:
            plt.hlines(self.threshold(), min(x), max(x), 'r')
        plt.draw()
        plt.pause(0.1)
        # if self.pf_intensities and plot_pf:
        #	_, (ax1, ax2) = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios':[2, 1], 'wspace':0.1}, num='stairs') # prepare a second panel for the pf plot
        #ax2.plot(self.pf_percent_correct, self.pf_intensities)
        # point_sizes = self.pf_responses_per_intensity * 5 # 5 pixels per trial at each point
        #ax2.scatter(self.pf_percent_correct, self.pf_intensities, s=point_sizes)
        #ax2.set_xlabel('Hit rate')
        # ax2.set_title('Psychometric\nfunction')

    def close_plot(self):
        plt.close('stairs')

    def _psychometric_function(self):
        """Create a psychometric function by binning data from a staircase
        procedure. Called automatically when staircase is finished. Sets
        pf_intensites
                        a numpy array of intensity values (where each is the center
                        of an intensity bin)
        pf_percent_correct
                        a numpy array of mean percent correct in each bin
        pf_responses_per_intensity
                        a numpy array of number of responses contributing to each mean
        """
        intensities = numpy.array(self.intensities)
        responses = numpy.array(self.data)
        binned_resp = []
        binned_intens = []
        n_points = []
        intensities = numpy.round(intensities, decimals=8)
        unique_intens = numpy.unique(intensities)
        for this_intens in unique_intens:
            these_resps = responses[intensities == this_intens]
            binned_intens.append(this_intens)
            binned_resp.append(numpy.mean(these_resps))
            n_points.append(len(these_resps))
        self.pf_intensities = binned_intens
        self.pf_percent_correct = binned_resp
        self.pf_responses_per_intensity = n_points


class Resultsfile():
    '''
    A class for simplifying the typical use cases of results files, including generating the name,
    creating the folders, and writing to the file after each trial. Writes a JSON Lines file,
    in which each line is a valid self-contained JSON string (http://jsonlines.org).
    Examples:
    >>> Resultsfile.results_folder = 'MyResults'
    >>> file = Resultsfile(subject='MS')
    >>> print(file.name)
    '''
    name = property(fget=lambda self: str(self.path.name), doc='The name of the results file.')

    def __init__(self, subject='test'):
        self.subject = subject
        self.path = Path(results_folder / Path(subject) / Path(subject +
                                                               datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S") + '.txt'))
        # make the Results folder and subject subfolder
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data, tag=None):
        '''
        Safely write data (must be json or convertable to json [string, list, dict, ...]) to the file.
        The file is opened just before writing and closed immediately after to avoid data loss.
        Call this method at the end of each trial to save the response and trial state.
        A tag (any string) can be prepended. If none is provided, the current time is used.
        '''
        try:
            data = json.loads(data) # if payload is already json, parse it into python object
        except (json.JSONDecodeError, TypeError):
            pass # if payload is not json - all good, will be encoded later
        if tag is None or tag == 'time':
            tag = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(self.path, 'a') as file:
            file.write(json.dumps({tag: data}))
            file.write('\n')

    @staticmethod
    def read_file(file_path, tag=None):
        '''
        Returns a list of objects saved in a results file at file_path.
        If a tag is given, then only the objects saved with that tag are returned.
        Objects are dictionaries with the tag as key and the saved data as value.
        '''
        content = []
        with open(file_path) as file:
            if tag is None:
                for line in file:
                    content.append(json.loads(line))
            else:
                for line in file:
                    jwd = json.loads(line)
                    if tag in jwd:
                        content.append(jwd[tag])
        return content

    def read(self, tag=None):
        '''
        Returns a list of objects saved in the current results file.
        If a tag is given, then only the objects saved with that tag are returned.
        Objects are dictionaries with the tag as key and the saved data as value.
        '''
        return Resultsfile.read_file(self.path, tag)

    def clear(self):
        'Clears the file by erasing all content.'
        with open(self.path, 'w') as file:
            file.write('')


class Precomputed(list):
    '''
    Class for randomly playing pre-computed sound stimuli without direct repetition.
    'sounds' can be a list of Sound objects, a function, or an iterable.
    This class simplifies generation and presentation of pre-computed sound stimuli
    and is typically used when stimulus generation takes too long to happen in each trial.
    In this case, a list of stimuli is precomputed and a random stimulus from the list is
    presented in each trial, ideally without direct repetition.
    The class allows easy generation of such stimulus lists (type 'slab.Precomputed') and
    keeps track of the previously presented stimulus. The list has a play method which
    automatically selects an element other than the previous one for playing, and
    can be used like an slab.Sound object.
    'sounds': [list|callable|iterator], the stimulus objects (must have a play method)
    'n': [int, *10*], only used if list is a callable, calls it n times to make the stimuli
    Examples:
    >>> stims = slab.Precomputed(sound_list) # using a pre-made list
    >>> stims = slab.Precomputed(lambda: slab.Sound.pinknoise(), n=10) # using a lambda function
    >>> # to make 10 examples of pink noise
    >>> stims = slab.Precomputed( (slab.Sound.vowel(vowel=v) for v in ['a','e','i']) ) # using a generator
    >>> stims.play() # playing a sound from the list
    '''

    def __init__(self, sounds, n=10):
        if isinstance(sounds, (list, tuple)):  # a list was passed, use as is
            list.__init__(self, sounds)
        elif callable(sounds):  # a function to generate sound objects was passed, call the function n times
            list.__init__(self, [])
            for _ in range(int(n)):
                list.append(self, sounds())
        elif isinstance(sounds, str):  # string is interpreted as name of a zip file containing the sounds
            with zipfile.ZipFile(sounds) as zip:
                files = zip.namelist()
                if files:
                    list.__init__(self, [])
                    for file in files:
                        list.append(self, slab.Sound(file))
        elif hasattr(sounds, '__iter__'):  # it's an iterable object, just iterate through it
            for sound in sounds:
                list.append(self, sound)
        else:
            raise TypeError('Unknown type for list argument.')
        self.previous = None  # this property holds the index of the previously played sound
        if not all(hasattr(sound, 'play') for sound in self):
            raise TypeError('Cannot play all of the provided items.')

    def play(self):
        idx = self.previous
        while idx == self.previous:
            idx = numpy.random.randint(len(self))
        self.previous = idx
        self[idx].play()

    def random_choice(self, n=1):
        'Return a random sample of sounds with replacement from the list.'
        idxs = numpy.random.randint(0, len(self), size=n)
        return [self[i] for i in idxs]

    def write(self, fname):
        fnames = list()
        for idx, sound in enumerate(self):
            f = f's_{idx}.wav'
            fnames.append(f)
            sound.write(f)
        with zipfile.ZipFile(fname, mode='w') as zip:
            for f in fnames:
                zip.write(f)


def load_config(config_file):
    '''
    Reads a text file with python varable assignments and returns
    a namedtuple with the variable names and values.
    Example:
    myconfig.txt:
    samplerate = 32000
    pause_duration = 30
    speeds = [60,120,180]
    >>> conf = load_config('myconfig.txt')
    >>> conf.speeds
    [60, 120, 180]
    '''
    from collections import namedtuple
    with open(config_file, 'r') as f:
        lines = f.readlines()
    if lines:
        var_names = []
        values = []
        for line in lines:
            var, val = line.strip().split('=')
            var_names.append(var.strip())
            values.append(eval(val.strip()))
        config_tuple = namedtuple('config', var_names)
        return config_tuple(*values)


if __name__ == '__main__':
    # Demonstration
    tr = Trialsequence(conditions=5, n_reps=2, name='test')
    stairs = Staircase(start_val=50, n_reversals=10, step_type='lin', step_sizes=[8, 4, 4, 2, 2, 1],
                       min_val=20, max_val=60, n_up=1, n_down=1, n_pretrials=4)
    for trial in stairs:
        response = stairs.simulate_response(thresh=30, transition_width=10)
        stairs.add_response(response)
        stairs.print_trial_info()
        stairs.plot()
    print(f'reversals: {stairs.reversal_intensities}')
    print(f'mean of final 6 reversals: {stairs.threshold()}')
