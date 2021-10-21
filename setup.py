from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dqn',
    version='0.1.2',
    description='Deep Q-Networks with Pytorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ilvieira/deep-q-networks',
    author='Inês Vieira',
    author_email='ines.l.vieira@tecnico.ulisboa.pt',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        "atari-py>=0.2.9",
        "gym>=0.15.7",
        "lz4>=3.1.3",
        "pandas>=1.3.3",
        "yaaf>=1.1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)