'''
psychoacoustics exports classes for handling psychophysical procedures and
measures, like trial sequences and staircases.
'''
import os
import io
import pathlib
import datetime
import json
import zipfile
import collections
from contextlib import contextmanager
from collections import Counter
try:
    import curses
    have_curses = True
except ImportError:
    have_curses = False
import numpy
try:
    import matplotlib.pyplot as plt
    have_pyplot = True
except ImportError:
    have_pyplot = False
import slab

results_folder = 'Results'
input_method = 'keyboard'  #: sets the input for the Key context manager to 'keyboard 'or 'buttonbox'

class _buttonbox:
    '''Adapter class to allow easy switching between input from the keyboard via curses and from the custom buttonbox adapter
    (arduino device that sends a keystroke followed by a return keystroke when pressing a button on the arduino).'''
    @staticmethod
    def getch():
        key = input() # buttonbox adapter has to return the keycode of intended keys!
        if key:
            return int(key)

@contextmanager
def Key():
    '''
    Wrapper for curses module to simplify getting a single keypress from the terminal (default) or a buttonbox.
    Set slab.psychoacoustics.input_method = 'buttonbox' to use a custom USB buttonbox.
    >>> with slab.Key() as key:
    >>>    response = key.getch()
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
        yield _buttonbox


class LoadSaveJson_mixin:
    'Mixin to provide JSON loading and saving functions'

    def save_json(self, file_name=None):
        """
        Save the object as JSON file.
        Arguments:
            file_name: name of the file to create or append. If `None`, returns an in-memory JSON object.
        """
        # self_copy = copy.deepcopy(self) use if reading the json file sometimes fails
        def default(i): return int(i) if isinstance(i, numpy.int64) else i
        if (file_name is None) or (file_name == 'stdout'):
            return json.dumps(self.__dict__, indent=2, default=default)
        try:
            with open(file_name, 'w') as f:
                json.dump(self.__dict__, f, indent=2, default=default)
                return True
        except OSError:
            return False

    def load_json(self, file_name):
        """
        Read JSON file and deserialize the object into self.__dict__.
        Attributes:
            file_name: name of the file to read.
        """
        with open(file_name, 'r') as f:
            self.__dict__ = json.load(f)


class TrialPresentationOptions_mixin:
    '''Mixin to provide alternative forced-choice (AFC) and Same-Different trial presentation methods and
    response simulation to Trialsequence and Staircase.'''

    def present_afc_trial(self, target, distractors, key_codes=(range(49, 58)), isi=0.25, print_info=True):
        '''
        Present the target sound in random order together with the distractor sound object (or list of
        several sounds) with isi pause (in seconds) in between, then aquire a response keypress via Key(), compare the
        response to the target interval and record the response via add_response. If key_codes for buttons are given
        (get with: ord('1') for instance -> ascii code of key 1 is 49), then these keys will be used as answer keys.
        Default are codes for buttons '1' to '9'. This is a convenience function for implementing alternative forced
        choice trials. In each trial, generate the target stimulus and distractors, then call present_afc_trial to play
        them and record the response. Optionally call print_trial_info afterwards.
        '''
        if isinstance(distractors, list):
            stims = [target] + distractors  # assuming sound object and list of sounds
        else:
            stims = [target, distractors]  # assuming two sound objects
        order = numpy.random.permutation(len(stims))
        for idx in order:
            stim = stims[idx]
            stim.play()
            plt.pause(isi)
        with Key() as key:
            response = key.getch()
        interval = numpy.where(order == 0)[0][0]
        interval_key = key_codes[interval]
        response = response == interval_key
        self.add_response(response)
        if print_info:
            self.print_trial_info()

    def present_tone_trial(self, stimulus, correct_key_idx=0, key_codes=(range(49, 58)), print_info=True):
        '''
        Present a stimulus and aquire a response. The response is compared with ``correct_key_idx`` (the index of the
        correct key in the ``key_codes`` argument) and a match is logged as True (correct response) or False (incorrect response).
        Arguments:
            stimulus: sound to present (object with play method)
            correct_key_idx: index of correct response key in ``key_codes``
            key_codes: list of response key codes (the default enables the number keys 1 to 9)
            print_info: if True, call print_trial_info
        '''
        stimulus.play()
        with slab.Key() as key:
            response = key.getch()
        response = response == key_codes[correct_key_idx]
        self.add_response(response)
        if print_info:
            self.print_trial_info()

    def simulate_response(self, threshold, transition_width=2, intervals=1, hitrates=None):
        '''Return a simulated response to the current condition index value by calculating the hitrate from a
        psychometric (logistic) function. This is only sensible if trials is numeric and an interval scale representing
        a continuous stimulus value.
        Arguments:
            thresh: midpoint/threshhold
            transition_width: range of stimulus intensities over which the hitrate increases from 0.25 to 0.75 (*2*)
            intervals: use 1 (default) to indicate a yes/no trial, 2 or more to indicate an AFC trial
            hitrates: list of hitrates for the different conditions, to allow custom rates instead of simulation.
                      If given, thresh and transition_width are not used.
        '''
        slope = 0.5 / transition_width
        if self.__class__.__name__ == 'Trialsequence': # check which class the mixin is in
            current_condition = self.trials[self.this_n]
        else:
            current_condition = self._next_intensity
        if hitrates is None:
            hitrate = 1 / (1 + numpy.exp(4 * slope  * (threshold - current_condition))) # scale/4  = slope at midpoint
        else:
            hitrate = hitrates[current_condition]
        hit = numpy.random.rand() < hitrate # True with probability hitrate
        if hit or intervals == 1:
            return hit
        return numpy.random.rand() < 1/intervals # still 1/intervals chance to hit the right interval


class Trialsequence(collections.abc.Iterator, LoadSaveJson_mixin, TrialPresentationOptions_mixin):
    """Non-adaptive trial sequences.
    Parameters:
        conditions: an integer, list, or flat array specifying condition indices,
            or a list of strings or other objects (dictionaries/tuples/namedtuples)
            specifying names or stimulus values for each condition.
            If given an integer x, uses range(x).
            If conditions is a string, then it is treated as the name of a previously
            saved trial sequence object, which is then loaded.
        n_reps: number of repeats of each condition (total trial number = len(conditions) * n_reps)
        trials: a list of conditions, i.e. the trial sequence. Typically, this list is left empty and generated by the
            class based on the other parameters.
        kind: The kind of sequence randomization used to generate the trial sequence. `non_repeating` (conditions are
            repeated in randome order `n_reps` times without direct repetition, default if n_conds > 2),
            `random_permutation` (conditions are permuted randomly without control over transition probabilities,
            (default if `n_conds` <= 2), or `infinite` (`non_repeating` if n_conds <= 2 or `random_permutation` trial
            sequence that reset when reaching the end to generate an infinite number of trials).
        label: a text label for the sequence.
    Attributes:
        .n_trials: the total number of trials that will be run
        .n_remaining: the total number of trials remaining
        .this_n: trial index in entire sequence, equals total trials completed so far
        .this_rep_n: index of repetition of the conditions we are currently in
        .this_trial_n: trial index within this repetition
        .this_trial: a dictionary giving the parameters of the current trial
        .finished: True/False: have we finished yet?
        .kind: records the kind of sequence (`random_permutation`, `non_repeating`, `infinite`)
"""
    def __init__(self, conditions=2, n_reps=1, deviant_freq=None, trials=None, kind=None, label=''):
        self.label = label
        self.n_reps = int(n_reps)
        self.conditions = conditions
        if isinstance(conditions, str) and os.path.isfile(conditions):
            self.load_json(conditions)  # import entire object from file
        else:
            if isinstance(conditions, int):
                self.conditions = list(range(1, conditions+1))
            else:
                self.conditions = conditions
            self.n_conds = len(self.conditions)
            self.trials = trials
            self.this_rep_n = 0  # index of repetition of the conditions we are currently in
            self.this_trial_n = -1  # trial index within this repetition
            self.this_n = -1  # trial index in entire sequence
            self.this_trial = []  # condition of current trial
            self.finished = False
            self.data = []  # holds responses if TrialPresentationOptions methods are called
            # generate stimulus sequence
            if self.trials is None:
                if kind is None:
                    kind = 'random_permutation' if self.n_conds <= 2 else 'non_repeating'
                if deviant_freq is not None:
                    deviants = slab.Trialsequence._deviant_indices(n_trials=int(self.n_conds * n_reps),
                                                                   deviant_freq=deviant_freq)
                if kind == 'random_permutation' or self.n_conds == 1:
                    trials = Trialsequence._create_random_permutation(self.n_conds, self.n_reps)
                elif kind == 'non_repeating':
                    trials = Trialsequence._create_simple_sequence(self.n_conds, self.n_reps)
                elif kind == 'infinite':
                    # implementation if infinite sequence is a bit of a hack (number of completed trials needs
                    # to be calculated as: trials.this_rep_n * trials.n_conds + trials.this_trial_n + 1)
                    # It's also not possible to make an infinite sequence with devaints.
                    if deviant_freq is not None:
                        raise ValueError("Deviants are not implemented for infinite sequences!")
                    if self.n_conds <= 2:
                        trials = Trialsequence._create_random_permutation(self.n_conds, 5)
                    else:
                        trials = Trialsequence._create_simple_sequence(self.n_conds, 1)
                else:
                    raise ValueError(f'Unknown kind parameter: {kind}!')
                if deviant_freq is not None:  # insert deviants
                    self.trials = list(numpy.insert(trials, deviants, 0))
                else:
                    self.trials = trials
            self.n_trials = len(self.trials)
            self.trials = [t.item() for t in self.trials]
            self.n_remaining = self.n_trials
            self.kind = kind
            self.data = [None for _ in self.trials]
            if deviant_freq is not None:
                self.n_conds += 1  # add one condition for deviants

    def __repr__(self):
        return self.__dict__.__repr__()

    def __str__(self):
        return f'Trialsequence, trials {"inf" if self.kind=="infinite" else self.n_trials}, remaining {"inf" if self.kind=="infinite" else self.n_remaining}, current condition {self.this_trial}'

    def __next__(self):
        """Advances to next trial and returns it. Updates attributes this_trial and this_trial_n.
        If the trials have ended this method will raise a StopIteration error.
        >>> trials = Trialsequence(.......)
        >>> for eachTrial in trials:  # automatically stops when done
        >>>     trials.print_trial_info()
        """
        self.this_trial_n += 1
        self.this_n += 1
        self.n_remaining -= 1
        if self.this_trial_n >= self.n_conds:  # start a new repetition
            self.this_trial_n = 0
            self.this_rep_n += 1
        if self.n_remaining < 0:  # all trials complete
            if self.kind == 'infinite':  # finite sequence -> reset and start again
                self.trials = Trialsequence._create_simple_sequence(
                    len(self.conditions), 1, previous=self.trials[-1])  # new sequence, avoid start with previous condition
                self.this_n = 0
                self.n_remaining = self.n_trials - 1  # reset trial countdown to length of new trial
                #  sequence (subtract 1 because we return the 0th trial below)
            else:  # finite sequence -> finish
                self.this_trial = []
                self.finished = True
        if self.finished:
            raise StopIteration
        if self.trials[self.this_n] == 0:
            self.this_trial = 0
        else:
            self.this_trial = self.conditions[self.trials[self.this_n]-1]  # fetch the trial info
        return self.this_trial

    def add_response(self, response):
        'Append a response value to the list `self.data`.'
        self.data[self.this_n] = response

    def print_trial_info(self):
        'Convenience method for printing current trial information.'
        print(f'{self.label} | trial # {self.this_n} of {"inf" if self.kind=="infinite" else self.n_trials} ({"inf" if self.kind=="infinite" else self.n_remaining} remaining): condition {self.this_trial}, last response: {self.data[-1] if self.data else None}')

    @staticmethod
    def _create_simple_sequence(n_conditions, n_reps, previous=1):
        '''Create a sequence of n_conditions x n_reps trials, where each repetitions contains all conditions in random
        order, and no condition is directly repeated across repetitions. `previous` can be set to an index in
        `range(n_conditions)``, which ensures that the sequence does not start with this index.'''
        permute = list(range(1, n_conditions+1))
        trials = [previous]
        for _ in range(n_reps):
            numpy.random.shuffle(permute)
            while trials[-1] == permute[0]:
                numpy.random.shuffle(permute)
            trials += permute
        trials = trials[1:]  # delete first entry ('previous')
        return trials

    @staticmethod
    def _deviant_indices(n_trials, deviant_freq=.1, mindist=3):
        '''Create sequence for an odball experiment which contains two conditions - standard (0) and
        deviant (1).

        Args:
            n_trials (int): length of the generated sequence.
            deviant_freq (float): frequency of the deviant, should not be greater than .25
            mindist (int): minimum number of standards between two deviants

        Returns:
            The return value. True for success, False otherwise.
        '''
        n_partials = int(numpy.ceil((2 / deviant_freq) - 7))
        reps = int(numpy.ceil(n_trials/n_partials))
        partials = []
        for i in range(n_partials):
            partials.append([1] * (mindist+i) + [0])
        idx = list(range(n_partials)) * reps
        numpy.random.shuffle(idx)
        trials = []
        for i in idx:  # make the trial sequence by putting possibilities together
            trials.extend(partials[i])
        trials = trials[:n_trials]  # cut the list to the requested number of trials
        return numpy.where([numpy.array(trials) == 0])[1]

    @staticmethod
    def _create_random_permutation(n_conditions, n_reps):
        '''Create a sequence of n_conditions x n_reps trials in random order.'''
        return list(numpy.random.permutation(numpy.tile(list(range(1, n_conditions+1)), n_reps)))

    def get_future_trial(self, n=1):
        """Returns the condition for n trials into the future or past,
        without advancing the trials. A negative n returns a previous (past)
        trial. Returns 'None' if attempting to go beyond the last trial.
        """
        if n > self.n_remaining or self.this_n + n < 0:
            return None
        return self.conditions[self.trials[self.this_n + n]]

    def transitions(self):
        'Return array (n_conds x n_conds) of transitions.'
        transitions = numpy.zeros((self.n_conds, self.n_conds))
        for i, j in zip(self.trials, self.trials[1:]):
            transitions[i, j] += 1
        return transitions

    def condition_probabilities(self):
        'Returns list of frequencies of conditions in the order listed in .conditions'
        probs = []
        for i in range(self.n_conds):
            num = self.trials.count(i)
            num /= self.n_trials
            probs.append(num)
        return probs

    def response_summary(self):
        '''Returns a tally of responses as list of lists for a finished Trialsequence.
        The indices of the outer list are the indices of the conditions in the sequence. Each inner list contains the
        number of responses per response key, with the response keys sorted in ascending order - the last element always
        represents None. For example, 3 conditions with 10 repetitions each, and 2 response keys (Yes/No experiment)
        + None returns a structure like this: [[0, 10, 0], [2, 8, 0], [9, 1, 0]], indicating that the person responded
        10 out of 10 times No in the first condition, 2 out of 10 Yes (and 8 out of 10 No) in the second,
        and 9 out of 10 Yes in the third condition. The third element in each sub-list is 0 meaning that there is no
        trial in which no response was given. These values can be used to construct hit rates and psychometric functions.
        '''
        if not self.finished:
            return None
        response_keys = list(set(self.data)) # list of used response key codes
        response_keys = sorted(response_keys, key=lambda x: (x is None, x))
        responses = []
        for condition in range(self.n_conds):
            idx = [i for i, cond in enumerate(self.trials) if cond == condition] # indices of condition in sequence
            count = Counter([self.data[i] for i in idx])
            resp_1cond = []
            for r in response_keys:
                resp_1cond.append(count[r])
            responses.append(resp_1cond)
        return responses

    def plot(self, axis=None, **kwargs):
        'Plot the trial sequence as scatter plot.'
        if not have_pyplot:
            raise ImportError('Plotting requires matplotlib!')
        if axis is None:
            axis = plt.subplot()
        axis.scatter(range(self.n_trials), self.trials, **kwargs)
        axis.set(title='Trial sequence', xlabel='Trials', ylabel='Condition index')
        plt.show()


class Staircase(collections.abc.Iterator, LoadSaveJson_mixin, TrialPresentationOptions_mixin):
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
    Attributes:
        .this_trial_n: number of completed trials
        .intensities: presented stimulus values
        .current_direction: 'up' or 'down'
        .data: list of responses
        .reversal_points: indices of reversal trials
        .reversal_intensities: stimulus values at the reversals (used to compute threshold)
        .finished: True/False: have we finished yet?
    """

    def __init__(self, start_val, n_reversals=None, step_sizes=1, step_up_factor=1, n_pretrials=0, n_up=1,
                 n_down=2, step_type='lin', min_val=-numpy.Inf, max_val=numpy.Inf, label=''):
        """
        Parameters
            label: A text label, printed by print_trial_info.
            start_val: initial stimulus value for the staircase
            n_reversals: number of reversals needed to terminate the staircase
            step_sizes: size of steps as a single value or a list/array. For a single value the step size is fixed.
                For a list/array the step size will progress to the next entry at each reversal.
                step_up_factor: allows different sizes for up and down steps to implement a Kaernbach1991 weighted
                up-down method. step_sizes sets down steps, which are multiplied by step_up_factor to obtain up step
                sizes. The default is 1, i.e. same size for up and down steps.
            n_pretrials: number of pretrials at start_val presented as familiarization before the actual experiment
            n_up: number of `incorrect` (or 0) responses before the staircase level increases
            n_down: number of `correct` (or 1) responses before the staircase level decreases
            step_type: `'lin', 'db', 'log'`. The type of steps that should be taken each time. 'lin' adds or subtracts
                that amount each step, 'db' and 'log' will step by a certain number of decibels or log units
                (prevents the stimulus value from reaching zero).
            min_val: smallest stimulus value permitted, or -Inf for staircase without lower limit
            max_val: largest stimulus value permitted, or Inf for staircase without upper limit
        """
        self.label = label
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
            if len(self.step_sizes) == 1:
                self.n_reversals = 8 # if Staircase called without parameters, construct a short 8-reversal test
            else:
                self.n_reversals = len(self.step_sizes) + 1 # otherwise dependend on number of stepsizes
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
        #if reversal and self._variable_step:  # new step size if necessary
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

    def threshold(self, n=0):
        '''Returns the average (arithmetic for step_type == 'lin',
        geometric otherwise) of the last n reversals (default: n_reversals - 1).'''
        if self.finished:
            if n == 0 or n > self.n_reversals:
                n = int(self.n_reversals) - 1
            if self.step_type == 'lin':
                return numpy.mean(self.reversal_intensities[-n:])
            return numpy.exp(numpy.mean(numpy.log(self.reversal_intensities[-n:])))
        return None # still running the staircase

    def print_trial_info(self):
        'Convenience method for printing current trial information.'
        print(
            f'{self.label} | trial # {self.this_trial_n}: reversals: {len(self.reversal_points)}/{self.n_reversals}, intensity {round(self.intensities[-1],2) if self.intensities else round(self._next_intensity,2)}, going {self.current_direction}, response {self.data[-1] if self.data else None}')

    def save_csv(self, fileName):
        'Write a csv text file with the stimulus values in the 1st line and the corresponding responses in the 2nd.'
        if self.this_trial_n < 1:
            return  # no trials to save
        with open(fileName, 'w') as f:
            raw_intens = str(self.intensities)
            raw_intens = raw_intens.replace('[', '').replace(']', '')
            f.write(raw_intens)
            f.write('\n')
            responses = str(numpy.multiply(self.data, 1))  # convert to 0 / 1
            responses = responses.replace('[', '').replace(']', '')
            responses = responses.replace(' ', ', ')
            f.write(responses)

    def plot(self, axis=None, **kwargs):
        'Plot the staircase. If called after each trial, one plot is created and updated.'
        if not have_pyplot:
            raise ImportError('Plotting requires matplotlib!')
        if self.intensities: # plotting only after first response
            x = numpy.arange(-self.n_pretrials, len(self.intensities)-self.n_pretrials)
            y = numpy.array(self.intensities) # all previously played intensities
            responses = numpy.array(self.data)
            if axis is None:
                fig = plt.figure('stairs')  # figure 'stairs' is created or made current
                axis = fig.gca()
            axis.clear()
            axis.plot(x, y, **kwargs)
            axis.set_xlim(-self.n_pretrials, max(20, (self.this_trial_n + 15)//10*10))
            axis.set_ylim(min(0, min(y)) if self.min_val == -numpy.Inf else self.min_val,
                        max(y) if self.max_val == numpy.Inf else self.max_val)
            # plot green dots at correct/yes responses
            axis.scatter(x[responses], y[responses], color='green')
            # plot red dots at correct/yes responses
            axis.scatter(x[~responses], y[~responses], color='red')
            axis.scatter(len(self.intensities)-self.n_pretrials+1, self._next_intensity, color='grey') # grey dot for current trial
            axis.set_ylabel('Dependent variable')
            axis.set_xlabel('Trial')
            axis.set_title('Staircase')
            if self.finished:
                axis.hlines(self.threshold(), min(x), max(x), 'r')
            plt.draw()
            plt.pause(0.01)

    def close_plot(self):
        'Closes a staircase plot (if not drawn into a specified axis).'
        plt.close('stairs')

    def _psychometric_function(self):
        """Create a psychometric function by binning data from a staircase procedure.
        Called automatically when staircase is finished. Sets attributes `pf_intensites` (array of intensity values
        where each is the center of an intensity bin), `pf_percent_correct` (array of mean percent correct in each bin),
        `pf_responses_per_intensity` (array of number of responses contributing to each mean).
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
    >>> Resultsfile.results_folder = 'MyResults'
    >>> file = Resultsfile(subject='MS')
    >>> print(file.name)
    '''
    name = property(fget=lambda self: self.path.name, doc='The name of the results file.')

    def __init__(self, subject='test'):
        self.subject = subject
        self.path = pathlib.Path(results_folder / pathlib.Path(subject) / pathlib.Path(subject +
                                    datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S") + '.txt'))
        # make the Results folder and subject subfolder
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data, tag=None):
        '''
        Safely write data (must be json or convertable to json [string, list, dict, ...]) to the file.
        The file is opened just before writing and closed immediately after to avoid data loss.
        Call this method at the end of each trial to save the response and trial state.
        A tag (any string) can be prepended. If None is provided, the current time is used.
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
        if len(content) == 1: # if only one item in list
            content = content[0] # get rid of the list
        return content

    def read(self, tag=None):
        '''
        Returns a list of objects saved in the current results file.
        If a tag is given, then only the objects saved with that tag are returned.
        Objects are dictionaries with the tag as key and the saved data as value.
        '''
        return Resultsfile.read_file(self.path, tag)

    @staticmethod
    def previous_file(subject=None):
        '''
        Returns the name of the most recently used resultsfile for a given `subject`.
        Intended for extracting information from a previous file when running partial experiments.
        '''
        path = pathlib.Path(results_folder) / pathlib.Path(subject)
        files = [f for f in path.glob(subject + '*') if f.is_file()]
        files.sort()
        return files[-1]

    def clear(self):
        'Clears the file by erasing all content.'
        with open(self.path, 'w') as file:
            file.write('')


class Precomputed(list):
    '''
    Class for randomly playing pre-computed sound stimuli without direct repetition.
    `sounds` can be a list of Sound objects, a function, or an iterable. This class simplifies generation and presentation
    of pre-computed sound stimuli and is typically used when stimulus generation takes too long to happen in each trial.
    In this case, a list of stimuli is precomputed and a random stimulus from the list is presented in each trial,
    ideally without direct repetition. The class allows easy generation of such stimulus lists (type `slab.Precomputed`)
    and keeps track of the previously presented stimulus. The list has a play method which automatically selects an
    element other than the previous one for playing, and can be used like an :meth:`slab.Sound` object.
    Attributes:
        sounds: sequence (list|callable|iterator) of stimulus objects (each must have a play method)
        n: only used if list is a callable, calls it n times to make the stimuli
    >>> stims = slab.Precomputed(sound_list) # using a pre-made list
    >>> stims = slab.Precomputed(lambda: slab.Sound.pinknoise(), n=10) # using a lambda function to make 10 examples of pink noise
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
        if not all(hasattr(sound, 'play') for sound in self):
            raise TypeError('Cannot play all of the provided items.') # all items in list need to have a play method
        self.sequence = [] # keep a list of indices of played stimuli, in case needed for later analysis

    def play(self):
        'Play a random, but never the previous, stimulus from the list.'
        if self.sequence:
            previous = self.sequence[-1]
        else:
            previous = None
        idx = previous
        while idx == previous:
            idx = numpy.random.randint(len(self))
        self.sequence.append(idx) # add to the list of played stimuli
        self[idx].play()

    def random_choice(self, n=1):
        'Returns a list of n random sounds with replacement.'
        idxs = numpy.random.randint(0, len(self), size=n)
        return [self[i] for i in idxs]

    def write(self, fname):
        'Writes the Precomputed object to disk as a zip file containing all sounds as wav files.'
        with zipfile.ZipFile(fname, mode='a') as zip_file: # open an empty zip file in 'append' mode
            for idx, sound in enumerate(self):
                f = io.BytesIO() # open a virtual file (file-like memory buffer)
                sound.write(f) # write sound to virtual file
                f.seek(0) # rewind the file so we can read it from start
                zip_file.writestr(f's_{idx}.wav', f.read())
                f.close()

    @staticmethod
    def read(fname):
        'Reads a zip file containing wav files and returns them as Precomputed object.'
        stims = Precomputed([])
        with zipfile.ZipFile(fname, 'r') as zip:
            files = zip.namelist()
            for file in files:
                wav_bytes = zip.read(file)
                stims.append(slab.Sound.read(io.BytesIO(wav_bytes)))
        return stims


def load_config(config_file):
    '''
    Reads a text file with python variable assignments and returns a namedtuple with the variable names and values.
    Contents of example.txt:
    >>> cat example.txt
    samplerate = 32000
    pause_duration = 30
    speeds = [60,120,180]
    Then call load_config to parse the file into a named tuple:
    >>> conf = load_config('example.txt')
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
    tr = Trialsequence(conditions=5, n_reps=2, label='test')
    stairs = Staircase(start_val=50, n_reversals=10, step_type='lin', step_sizes=[8, 4, 4, 2, 2, 1],
                       min_val=20, max_val=60, n_up=1, n_down=1, n_pretrials=4)
    for trial in stairs:
        response = stairs.simulate_response(threshold=30, transition_width=10)
        stairs.add_response(response)
        stairs.print_trial_info()
        stairs.plot()
    print(f'reversals: {stairs.reversal_intensities}')
    print(f'mean of reversals: {stairs.threshold()}')
