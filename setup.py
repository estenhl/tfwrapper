from setuptools import setup, find_packages

setup(
    name='tfwrapper',
    version='0.1.0-rc3',
    description='Wrapper for tensorflow',
    url='https://github.com/epigramai/tfwrapper',
    author='Esten Høyland Leonardsen',
    author_email='esten@epigram.ai',
    packages=find_packages('.'),
    install_requires=['tensorflow>=1.0.0']
)
