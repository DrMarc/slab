import slab
from itertools import zip_longest


def test_trialsequence():
    seq = slab.Trialsequence(conditions=5, n_reps=10, kind="random_permutation")
    inf = slab.Trialsequence(conditions=5, kind="infinite")
    for trial in seq:
        inf.__next__()
    seq = slab.Trialsequence(conditions=5, n_reps=10, kind="non_repeating", deviant_freq=0.1)
    n_deviants = len([i for i, cond in enumerate(seq.trials) if cond == 0])
    seq.transitions()
    seq.condition_probabilities()
    for trial in seq:
        if trial != 0:
            seq.add_response(seq.simulate_response(threshold=1))
    assert seq.data.count(None) == n_deviants


def test_staircase():
    # normal staircase
    stairs1 = slab.Staircase(start_val=10, n_reversals=4)
    stairs2 = slab.Staircase(start_val=8, n_reversals=6)
    stairs = zip_longest(stairs1, stairs2)
    for stim1, stim2 in stairs:
        if stim1:
            r1 = stairs1.simulate_response(4)
            stairs1.add_response(r1)
            # stairs1.print_trial_info()
        if stim2:
            r2 = stairs2.simulate_response(2)
            stairs2.add_response(r2)
            # stairs2.print_trial_info()
    stairs1.plot()
    stairs2.plot()
    # adaptive staircase
    stairs = slab.Staircase(start_val=10, n_reversals=18, step_sizes=[4, 1])
    for stimulus_value in stairs:
        response = stairs.simulate_response(threshold=3)
        stairs.add_response(response)
    stairs.save_csv("staircase.csv")


def test_readwrite():

    sounds = [slab.Sound.whitenoise() for _ in range(10)]
    sounds = slab.Precomputed(sounds)
    sounds
    pass

# Functions to test:

# present_afc_trial
# present_tone_trial
# save_csv
# class Resultsfile
# class Precomputed
# load_config
