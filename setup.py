from setuptools import setup, find_packages

setup(
    name='warehouse',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['gym==0.25.0', 'pygame==2.1.0']
)