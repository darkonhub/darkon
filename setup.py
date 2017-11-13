from setuptools import setup, find_packages
import pypandoc

version = '0.0.1'
github_url = 'https://github.com/darkonhub/darkon'

with open('requirements.txt') as f:
    deps = [str(dep.strip()) for dep in f.readlines()]

setup(
    name='darkon',
    packages=find_packages('.'),
    version=version,
    description='hack your deep learning performance ',
    long_description=pypandoc.convert('README.md', 'rst'),
    author='Neosapience Inc.',
    url=github_url,
    download_url='{}/tarball/{}'.format(github_url, version),
    keywords=['AI', 'ML', 'DL', 'deep learning', 'machine learning', 'debugging',
              'hack', 'performance', 'tuning', 'tensorflow', 'tf'],
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Framework :: Tensorflow',
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
