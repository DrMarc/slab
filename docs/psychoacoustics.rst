.. _Psychoacoustics:

Psychoacoustics
===============

Trial sequences
---------------

Adaptive staircases
-------------------

Precomputed sounds
------------------

Other useful functions
----------------------

**Psychoacoustics**: A collection of classes for working trial sequences, adaptive staircases, forced-choice procedures, stimulus presentation and response recording from the keyboard and USB button boxes, handling of precomputed stimulus lists, results files, and experiment configuration files.::

    # set up an 1up-2down adaptive weighted staircase with dynamic step sizes:
    stairs = slab.Staircase(start_val=10, max_val=40, n_up=1, n_down=2, step_sizes=[3, 1], step_up_factor=1.5)
    for trial in stairs: # draw a value from the staircase; the loop terminates with the staircase
        response = stairs.simulate_response(30) # simulate a response from a participant using a psychometric function
        print(f'trial # {stairs.this_trial_n}: intensity {trial}, response {response}')
        stairs.add_response(response) # logs the response and advances the staircase
		stairs.plot() # updates a plot of the staircase in each trial to keep an eye on the performance of the listener
    stairs.reversal_intensities # returns a list of stimulus values at the reversal points of the staircase
    stairs.threshold() # computes and returns the final threshold
    stairs.save_json('stairs.json') # the staircase object can be saved as a human readable json file

    # for non-adaptive experiments and all other cases where you need a controlled sequence of stimulus values:
    trials = slab.Trialsequence(conditions=5, n_reps=2) # sequence of 5 conditions, repeated twice, without direct repetitions
    trials = slab.Trialsequence(conditions=['red', 'green', 'blue'], kind='infinite') # infinite sequence of color names
    trials = slab.Trialsequence.mmn_sequence(n_trials=60, deviant_freq=0.12) # stimulus sequence for an oddball design
    trials.transitions() # return the array of transition probabilities between all combinations of conditions.
    trials.condition_probabilities() # return a list of frequencies of conditions
    for trial in trials: # use the trials object in a loop to go through the trials
        print(trial) # here you would generate or select a stimulus according to the condition
        trials.present_afc_trial(target, distractor, isi=0.2) # present a 2-alternative forced-choice trial and record the response

    stims = slab.Precomputed(lambda: slab.Sound.pinknoise(), n=10) # make 10 instances of noise as one Sound-like object
    stims = slab.Precomputed([stim1, stim2, stim3, stim4, stim5]) # or use a list of sound objects, or a list comprehension
    stims.play() # play a random instance
    stims.play() # play another one, guaranteed to be different from the previous one
	stims.sequence # the sequence of instances played so far
    stims.save('stims.zip') # save the sounds as zip file
    stims = slab.Precomputed.read('stims.zip') # reloads the file into a Precomputed object
