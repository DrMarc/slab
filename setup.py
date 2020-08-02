from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('VERSION.txt') as f:
    version = f.read().strip()

setup(name='soundtools',
	version=version,
	description='Tools for generating and manipulating digital signals, particularly sounds.',
	long_description=readme,
	long_description_content_type='text/markdown',
	url='http://github.com/DrMarc/soundtools.git',
	author='Marc Schoenwiesner',
	author_email='marc.schoenwiesner@gmail.com',
	license='MIT',
	python_requires='>=3.6',
	packages=['slab'],
	package_data={'slab': ['data/mit_kemar_normal_pinna.sofa', 'data/KEMAR_interaural_level_spectrum.npy']},
	include_package_data=True,
	zip_safe=False)
