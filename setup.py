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
	package_data={'slab': ['data/*.sofa', 'data/*.npy']},
	include_package_data=True,
	zip_safe=False)
