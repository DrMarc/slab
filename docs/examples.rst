
.. currentmodule:: slab.experiments

Worked examples
===============

The folder slab.experiments contains the full code from actual psychoacoustic experiments in our lab. We use this folder mainly to make the code available and enable easy replication. The examples are well documented and may give you and idea of the typical structure of such experiments. To run these experiments, import them from slab.experiments::

    from slab.experiments import motion_speed
    motion_speed.main_experiment(subject='test')

Currently available are:

.. autofunction:: slab.experiments.room_voice_interference.main_experiment

.. autofunction:: slab.experiments.motion_speed.main_experiment


Quick standard experiments:
---------------------------

.. audiogram:

Audiogram
^^^^^^^^^
Run a pure tone audiogram at the standard frequencies 125, 250, 500, 1000, 2000, 4000 Hz using an adaptive staircase: ::

    from matplotlib import pyplot as plt
    freqs = [125, 250, 500, 1000, 2000, 4000]
    threshs = []
    for frequency in freqs:
        stimulus = slab.Sound.tone(frequency=frequency, duration=0.5)
        stairs = slab.Staircase(start_val=50, n_reversals=18)
        print(f'Starting staircase with {frequency} Hz:')
        for level in stairs:
            stimulus.level = level
            stairs.present_tone_trial(stimulus)
            stairs.print_trial_info()
        threshs.append({stairs.threshold())
        print(f'Threshold at {frequency} Hz: {stairs.threshold()} dB')
    plt.plot(freqs, threshs) # plot the audiogram


Temporal modulation transfer function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Measure temporal modulation transfer functions via detection thresholds for amplitude modulations. The parameters of the test replicate Fig. 2 in Viemeister [1979]_ and present sinusoidal 2 to 4000 Hz modulations in a 77-dB wideband noise carrier using an adaptive staircase. ::

    from matplotlib import pyplot as plt
    mod_freqs = [2, 4, 8, 16, 32, 64, 125, 250, 500, 1000, 2000, 4000]
    threshs = []
    base_stimulus = slab.Sound.pinknoise(duration=1.)
    base_stimulus.level = 77
    for frequency in mod_freqs:
    stairs = slab.Staircase(start_val=0.8, n_reversals=16, step_type='db',
                step_sizes=[4,2], min_val=0, max_val=1, nup=1, ndown=2)
        print(f'Starting staircase with {frequency} Hz:')
        for depth in stairs:
            stimulus = base_stimulus.am(frequency=frequency, depth=depth)
            stairs.present_afc_trial(stimulus, base_stimulus)
        threshs.append(stairs.threshold(n=14))
        print(f'Threshold at {frequency} Hz: {stairs.threshold(n=14)} modulation depth')
    plt.plot(freqs, threshs) # plot the transfer function


.. [1979] Viemeister (1979) Temporal modulation transfer functions based upon modulation thresholds. JASA 66(5), 1364–1380
