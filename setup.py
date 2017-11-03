#!/usr/bin/env python3
from setuptools import setup

long_description = "Python TIFF Stack Reader for ScanImage 5 scans (including multiROI)."

setup(
    name='scanreader',
    version='0.4.0',
    description="Reader for ScanImage 5 scans (including multiROI).",
    long_description=long_description,
    author='Erick Cobos',
    author_email='ecobos@bcm.edu',
    license='MIT',
    url='https://github.com/atlab/scanreader',
    keywords='ScanImage scanreader multiROI 2016b tiff',
    packages=['scanreader'],
    install_requires=['numpy>=1.12.0', 'tifffile>=0.12.0'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English'
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
