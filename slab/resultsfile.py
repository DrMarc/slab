'''
psychoacoustics exports classes for handling psychophysical procedures and
measures, like trial sequences and staircases.
This module uses doctests. Use like so:
python -m doctest psychoacoustics.py
'''
from pathlib import Path
import datetime

results_folder = 'Results'

class Resultsfile():
	'''
	A class for simplifying the typical use cases of results files, including generating the name,
	creating the folders, and writing to the file after each trial.
	Examples:
	>>> Resultsfile.results_folder = 'MyResults'
	>>> file = Resultsfile(subject='MS')
	>>> print(file.name)
	'''

	name = property(fget=lambda self: str(self.path.name), doc='The name of the results file.')

	def __init__(self, subject='test'):
		self.subject = subject
		self.path = Path(results_folder / Path(subject) / Path(subject + datetime.datetime.now().strftime("_%Y-%m-%d-%H-%M-%S") + '.txt'))
		self.path.parent.mkdir(parents=True, exist_ok=True) # make the Results folder and subject subfolder

	def write(self, data):
		'''
		Safely write data (must be lines of text or convertable to lines of text) to the file.
		The file is opened just before writing and closed immediately after to avoid data loss.
		Call this method at the end of each trial to save the response and trial state.
		'''
		# convert to text if necessary
		with open(self.path, 'a') as file:
			file.writelines(data)
