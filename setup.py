from setuptools import setup

setup(
    name='tfwrapper',
    version='0.0.2',
    description='Wrapper for tfwrapper',
    url='https://github.com/epigramai/tfwrapper',
    author='Esten Høyland Leonardsen',
    author_email='esten@epigram.ai',
    # license='<license>',
    packages=['tfwrapper'],
    install_requires=['tensorflow>=0.11.0'],
)
