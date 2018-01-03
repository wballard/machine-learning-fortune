'''
Just the setup.
'''


from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

# Package details
setup(
    name='machine-learning-fortune',
    version='0.0.1',
    author='Will Ballard',
    author_email='wballard@mailframe.net',
    url='https://github.com/wballard/machine-learning-fortune',
    description='Make a `fortune` with machine learning!',
    long_description=readme(),
    license='BSD 3-Clause License',
    packages=find_packages(),
    scripts=['bin/machine-learning-fortune'],
    install_requires=[
        'keras>=2.1.2',
        'tensorflow>=1.4.1',
        'numpy>=1.13.1',
        'tqdm',
        'docopt',
        'scikit-learn>=0.18.1'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)