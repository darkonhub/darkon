from setuptools import setup, find_packages
from m2r import parse_from_file
import json


info = json.load(open('info.json'))
with open('requirements.txt') as f:
    deps = [str(dep.strip()) for dep in f.readlines()]

setup(
    name=info['name'],
    packages=find_packages('.'),
    version=info['version'],
    description='Performance hacking for your deep learning models',
    long_description=parse_from_file('README.md', encoding='utf-8'),
    author=info['authors'],
    url=info['github_url'],
    download_url='{}/tarball/{}'.format(info['github_url'], info['version']),
    keywords=['AI', 'ML', 'DL', 'deep learning', 'machine learning', 'neural network',
              'deep neural network', 'debug neural networks', 'performance hacking',
              'tensorflow', 'tf'],
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Debuggers',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    install_requires=deps
)
