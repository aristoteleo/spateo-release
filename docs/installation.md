# Installation

## Prerequisites

Make sure you have at least Python 3.7 installed. Spateo does not officially support earlier Python versions.

## Installation

Spateo can be installed either by downloading a release uploaded to the Python Package Index (PyPI), or the most up-to-date version directly from our GitHub repository.

### PyPI

The following command will download and install the most recent release of Spateo.

```
pip install spateo-release
```

### GitHub

To have access to the most up-to-date version (which may include features not yet in the PyPI version), Spateo can be installed directly from the `main` branch of our GitHub repository.

```
pip install git+https://github.com/aristoteleo/spateo-release
```

To install Spateo from a specific GitHub branch,

```
pip install git+https://github.com/aristoteleo/spateo-release@{branch}
```

where `{branch}` is the branch name.

## Known Issues and Fixes

There sometimes may be issues reading datasets from AnnData objects. To remedy this, manually install the following 
package versions into the environment that Spateo is contained in (if these versions are not already installed):

```
pip install h5py==3.7.0
pip install anndata==0.8.0
```

## Development

If you are interested in contributing to Spateo, please read [](contributing).
