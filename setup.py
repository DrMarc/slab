from setuptools import setup, find_packages
import re

with open('README.md') as f:
    readme = f.read()

# extract version
with open('slab/__init__.py') as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setup(name='slab',
      version=version,
      description='Tools for generating and manipulating digital signals, particularly sounds.',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://github.com/DrMarc/slab.git',
      author='Marc Schoenwiesner',
      author_email='marc.schoenwiesner@gmail.com',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['numpy', 'scipy', 'SoundFile', 'SoundDevice',
        "matplotlib < 3.4; python_version == '3.6'",
        "matplotlib; python_version >= '3.7'"],
      extras_require={'testing': ['pytest', 'h5netcdf'],
                      'docs': ['sphinx', 'sphinx-rtd-theme'],
                      'hrtf': ['h5netcdf']},
      packages=find_packages(),
      package_data={'slab': ['data/mit_kemar_normal_pinna.bz2']},
      include_package_data=True,
      zip_safe=False)
