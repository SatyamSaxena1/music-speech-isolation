from setuptools import setup, find_packages

setup(
    name='speech_isolation',
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[],
    description='Speech isolation scaffold',
)
