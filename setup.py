#!/usr/bin/env python
import os
import sys
from setuptools import setup
import chesslib

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit(0)


classifiers = '''\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Operating System :: OS Independent
Topic :: Software Development :: Libraries :: Python Modules
'''.splitlines()

with open('README.rst') as fp:
    long_description = fp.read()


setup(
    name='chesslib',
    version=chesslib.get_version(),
    author='David Krauth',
    author_email='dakrauth@gmail.com',
    url='https://github.com/dakrauth/chesslib',
    license='MIT',
    platforms=['any'],
    py_modules=['chesslib'],
    description=chesslib.__doc__,
    classifiers=classifiers,
    long_description=long_description
)