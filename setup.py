import os
import sys

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    c_opts = {
        "unix": ["-O3", "-std=c++11", "-fPIC"],
        "msvc": ["/EHsc"],
    }

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == "unix":
            if sys.platform == "darwin":
                opts += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


# Relative path to C++ source files
fastpd_src = [
    # Corresponds to the source code under external/fastpd/fastpd/
    os.path.join("external", "fastpd", "fastpd", "FastPD.cpp"),
    os.path.join("external", "fastpd", "fastpd", "graph.cpp"),
    # Plus the module initialization file
    os.path.join("external", "fastpd", "init_fastpd.cpp"),
]

# Include search paths for header files
include_dirs = [
    # PyBind11's headers
    pybind11.get_include(),
    # The directory where FastPD's own header files are located
    os.path.join("external", "fastpd", "fastpd"),
    # If FastPD uses other custom includes, add them here
]

extensions = [
    Extension(
        name="spateo.libfastpd.libfastpd",  # This is the final module name in Python
        sources=fastpd_src,
        include_dirs=include_dirs,
        language="c++",
        # libraries=[] If you need to link to Python3 itself, you usually don't need to write it explicitly, because PyBind11 will handle it implicitly
        # library_dirs=[] If there are additional library paths, write them here as well
    ),
]


def read_requirements(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if not line.isspace()]


with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(
    name="spateo-release",
    version="1.1.1",
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
    description="Spatiotemporal modeling of molecular holograms",
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
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
