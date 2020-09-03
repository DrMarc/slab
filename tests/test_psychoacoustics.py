import slab
import numpy

def test_trialsequence():

    seq = slab.Trialsequence(conditions=5, kind="infinite")
    for i in range(10):
        seq.__next__()
        seq.get_future_trial(i)

    kinds = ["non_repeating", "random_permutation"]
    for kind in kinds:
        seq = slab.Trialsequence(conditions=5, n_reps=10, kind=kind)
        seq.transitions()
        seq.condition_probabilities()
        for trial in seq:
            seq.add_response(seq.simulate_response(threshold=1))
        assert seq.response_summary() is not None
        seq.plot()

    deviant_freq = 0.15
    seq = slab.Trialsequence(conditions=1, n_reps=1000,
        deviant_freq=deviant_freq, kind="mismatch_negativity")
    sum(seq.trials)/len(seq.trials)


def test_mmnsequence():
    deviant_freq = 0.15
    n_trials = 100
    seq = slab.Trialsequence.mmn_sequence(n_trials=110, deviant_freq=deviant_freq)
    sum(seq.trials)

    n_partials = int(numpy.ceil((2 / deviant_freq) - 7))
    reps = int(numpy.ceil(n_trials/n_partials))

    partials = []
    for i in range(n_partials):
        partials.append([0] * (3+i) + [1])
    print(partials)

    idx = list(range(n_partials)) * reps
    numpy.random.shuffle(idx)


# def test_keyinput():
# Seems like curses does not run in pytest - is there a workaround?
#    with slab.Key() as key:
#        response = key.getch()
#    print(response)
#    seq.present_afc_trial(target=slab.Sound.tone(), distractors=slab.Sound.whitenoise())
