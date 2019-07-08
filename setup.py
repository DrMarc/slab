from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='slab',
	version='0.5',
	description='Tools for generating and manipulating digital signals, particularly sounds.',
	url='http://github.com/',
	author='Marc Schoenwiesner',
	author_email='marc.schoenwiesner@gmail.com',
	license='MIT',
	packages=['slab'],
	include_package_data=True,
	zip_safe=False)
