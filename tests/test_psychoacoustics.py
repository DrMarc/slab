import slab
import curses
"""
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
# yield stdscr
curses.nocbreak()
curses.echo()
curses.endwin()
stdscr.keypad(True)
"""
def test_trialsequence():

    seq = slab.Trialsequence(conditions=5, n_reps=100, kind="non_repeating")

# def test_keyinput():
#    seq.present_afc_trial(target=slab.Sound.tone(), distractors=slab.Sound.whitenoise())

#    with slab.Key() as key:
#        response = key.getch()  # how to test this if there is no key input?
#    slab.psychoacoustics.input_method = 'buttonbox'
