import os

from setuptools import find_packages, setup


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spateo-release",
    version="0.0.0",
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "docs": read_requirements(os.path.join("docs", "requirements.txt")),
    },
    packages=find_packages(exclude=("tests", "docs")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    author="Xiaojie Qiu",
    author_email="xqiu.sc@gmail.com",
    description="A complete solution of spatialtemporal dynamics analyses toolkit of single cell spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    url="https://github.com/aristoteleo/spat-release",
    download_url=f"https://github.com/aristoteleo/spatt-release",
    keywords=[
        "SpatialTranscriptomics",
        "stereo-seq",
        "Visium",
        "seqFish",
        "MERFISH",
        "slide-seq",
        "DBiT-seq",
        "HDST-seq",
        "osmFISH",
        "spatialtemporal",
    ],
)
