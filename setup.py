import os

from setuptools import find_packages, setup


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(
    name="spateo-release",
    version="1.0.2",
    python_requires=">=3.7",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "docs": read_requirements(os.path.join("docs", "requirements.txt")),
        "3d": read_requirements("3d-requirements.txt"),
    },
    packages=find_packages(exclude=("tests", "docs")),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    author="Xiaojie Qiu",
    author_email="xqiu@wi.mit.edu",
    description="Spateo: multidimensional spatiotemporal modeling of single-cell spatial transcriptomics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD",
    url="https://github.com/aristoteleo/spateo-release",
    keywords=[
        "spatial-transcriptomics",
        "stereo-seq",
        "Visium",
        "seqFish",
        "MERFISH",
        "slide-seq",
        "DBiT-seq",
        "HDST-seq",
        "osmFISH",
        "spatiotemporal",
    ],
)
