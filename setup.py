from setuptools import setup

setup(
    name="smelu",
    version="0.1",
    url="https://github.com/ChristophReich1996/SmeLU",
    license="MIT License",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="PyTorch SmeLU",
    packages=["smelu",],
    install_requires=["torch>=1.0.0"],
)