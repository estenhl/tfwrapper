from setuptools import setup, find_packages

setup(
    name='tfwrapper',
    version='0.0.4',
    description='Wrapper for tfwrapper',
    url='https://github.com/epigramai/tfwrapper',
    author='Esten Høyland Leonardsen',
    author_email='esten@epigram.ai',
    # license='<license>',
    # packages=['tfwrapper', 'tfwrapper.utils'],
    packages=find_packages('.'),
    install_requires=['tensorflow>=1.0.0'],
)
