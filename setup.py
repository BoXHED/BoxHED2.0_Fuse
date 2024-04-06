from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name='BoXHED_Fuse',
    version='1.0.0',
    description='',
    author='Aaron Su',
    author_email='aa_ron_su@tamu.edu',
    install_requires=install_requires,
    packages=find_packages(),
)


