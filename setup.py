from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

# extract version
with open('_version.py') as f:
    version_file_content = f.read().strip()

pattern = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(pattern, version_file_content, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in _version.py')


setup(name='soundlab',
      version=version,
      description='Tools for generating and manipulating digital signals, particularly sounds.',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://github.com/DrMarc/soundlab.git',
      author='Marc Schoenwiesner',
      author_email='marc.schoenwiesner@gmail.com',
      license='MIT',
      python_requires='>=3.6',
      packages=find_packages(),
      package_data={'slab': ['data/mit_kemar_normal_pinna.sofa',
                             'data/KEMAR_interaural_level_spectrum.npy']},
      include_package_data=True,
      zip_safe=False)
