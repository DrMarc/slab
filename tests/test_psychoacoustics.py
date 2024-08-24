import slab
from itertools import zip_longest
import tempfile
import numpy
from matplotlib import pyplot as plt
import random
from pathlib import Path
DIR = tempfile.TemporaryDirectory()
PATH = Path(DIR.name)
plt.ioff()
# NOTE: Everything involving pressing a key is currently untested because curses does not run within pytest


def test_sequence_from_trials():
    for i in range(100):  # generate from integers
        start = numpy.random.randint(1, 100)
        stop = start+numpy.random.randint(1, 10)
        trials = [numpy.random.randint(start, stop) for i in range(100)]
        sequence = slab.Trialsequence(trials=trials)
        assert all(numpy.unique(sequence.trials) == numpy.array(range(1, sequence.n_conditions+1)))
    trials = ["a", "x", "x", "z", "a", "a", "a", "z"]
    sequence = slab.Trialsequence(trials=trials)
    assert all(numpy.unique(sequence.trials) == numpy.array(range(1, sequence.n_conditions + 1)))
    sounds = [slab.Sound.pinknoise(), slab.Sound.whitenoise()]
    trials = [random.choice(sounds) for i in range(50)]
    sequence = slab.Trialsequence(trials=trials)
    assert all(numpy.unique(sequence.trials) == numpy.array(range(1, sequence.n_conditions + 1)))


def test_sequence():
    for _ in range(100):
        conditions_list = [numpy.random.randint(2, 10), ["a", "b", "c"], [("a", "b"), (1.5, 3.2)],
                           [slab.Sound.pinknoise(), slab.Sound.whitenoise()]]
        kinds = ["random_permutation", "non_repeating", "infinite"]
        for conditions in conditions_list:
            n_reps = numpy.random.randint(1, 10)
            kind = random.choice(kinds)
            sequence = slab.Trialsequence(conditions=conditions, n_reps=n_reps, kind=kind)
            if isinstance(conditions, int):
                conditions = list(range(1, conditions+1))
            assert sequence.conditions == conditions
            assert all(numpy.unique(sequence.trials) == numpy.array(range(1, sequence.n_conditions + 1)))
            if kind != "infinite":
                assert sequence.n_trials == len(conditions) * n_reps
                for trial in sequence:
                    assert trial == sequence.conditions[sequence.trials[sequence.this_n]-1]
            else:
                count = 0
                for trial in sequence:
                    assert trial == sequence.conditions[sequence.trials[sequence.this_n]-1]
                    count += 1
                    if count > 100:
                        break


def test_deviants():
    for i in range(100):
        conditions = 4
        n_reps = 50
        n_trials = conditions*n_reps
        deviant_frequency = 0.2 * random.random() + .05
        sequence = slab.Trialsequence(conditions=4, n_reps=50, deviant_freq=deviant_frequency)
        count_deviants = 0
        for trial in sequence:
            if trial == 0:
                count_deviants += 1
            else:
                assert trial == sequence.conditions[sequence.trials[sequence.this_n] - 1]
        assert count_deviants == sequence.trials.count(0) == int(n_trials*deviant_frequency)


def test_staircase():  # this seems to block
    stairs1 = slab.Staircase(start_val=10, n_reversals=4)
    stairs2 = slab.Staircase(start_val=8, n_reversals=6)
    stairs = zip_longest(stairs1, stairs2)
    count = 0
    for stim1, stim2 in stairs:
        count += 1
        if count > 100:
            break
        print(count)
        if stim1 is not None:
            r1 = stairs1.simulate_response(4)
            stairs1.add_response(r1)
            # stairs1.print_trial_info()
        if stim2 is not None:
            r2 = stairs2.simulate_response(2)
            stairs2.add_response(r2)
            # stairs2.print_trial_info()
    fig, ax = plt.subplots(2)
    stairs1.plot(axis=ax[0], show=False)
    stairs2.plot(axis=ax[1], show=False)
    # adaptive staircase
    stairs = slab.Staircase(start_val=10, n_reversals=18, step_sizes=[4, 1])
    for stimulus_value in stairs:
        response = stairs.simulate_response(threshold=3)
        stairs.add_response(response)
    stairs.save_csv(PATH / "staircase.csv")


def test_precomputed():
    sounds = [slab.Sound.whitenoise() for _ in range(10)]
    sounds = slab.Precomputed(sounds)
    sounds = slab.Precomputed(sounds.random_choice(5))
    sounds.write(PATH / "precomputed.zip")
    sounds = slab.Precomputed.read(PATH / "precomputed.zip")


def test_results():
    slab.psychoacoustics.results_folder = PATH
    results = slab.ResultsFile(subject="MrPink")
    for data in [[1, 2, 3], slab.Trialsequence()]:
        results.write(data)
    results.read()
    results = slab.ResultsFile.read_file(slab.ResultsFile.previous_file(subject="MrPink"))
    results.clear()
    # ResultsTable
    results = slab.ResultsTable(subject="MrPink", columns="subject, trial")
    row = results.Row(subject=results.subject, trial=1)
    results.write(row)
