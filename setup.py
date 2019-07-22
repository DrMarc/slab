from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='slab',
	version='0.6',
	description='Tools for generating and manipulating digital signals, particularly sounds.',
	url='http://github.com/DrMarc/soundtools.git',
	author='Marc Schoenwiesner',
	author_email='marc.schoenwiesner@gmail.com',
	license='MIT',
	packages=['slab'],
	data_files=[('data', ['data/mit_kemar_normal_pinna.sofa', 'data/KEMAR_interaural_level_spectrum.npy'])],
	include_package_data=True,
	zip_safe=False)
