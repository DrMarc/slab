Worked examples
===============

.. _audiogram:
Audiogram
---------

run a pure tone audiogram using an adaptive staircase: ::

  import slab
  stimulus = slab.Sound.tone(frequency=500, duration=0.5) # make a 0.5 sec pure tone of 500 Hz
  stairs = slab.Staircase(start_val=50, n_reversals=10) # set up the adaptive staircase
  for level in stairs: # the staircase object returns a value between 0 and 50 dB for each trial
      stimulus.level = level
      stairs.present_tone_trial(stimulus) # plays the tone and records a keypress (1 for 'heard', 2 for 'not heard')
      stairs.print_trial_info() # optionally print information about the current state of the staircase
  print(stairs.threshold()) # print threshold when done
