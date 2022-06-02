```{highlight} shell
```

# Contributing

Contributions are always welcome!

## Prerequisites

1. Make a local clone of the Spateo repository.

2. Install Python requirements for development.

    ```
    pip install -r dev-requirements.txt
    ```

3. Install pre-commit hooks.

    We use pre-commit to enforce code quality standards. Specifically, we use
    `black` for code formatting and `isort` for import formatting.

    ```
    pre-commit install
    ```

    Once the hooks are installed, your code will automatically be re-formatted
    every time you make a commit.

## Testing

We use `nosetests` for unit-testing. All tests are located in the `tests` directory.
Whenever you make a change (whether it may be a bug fix, new feature, etc.), you
should run unit-tests to make sure everything is working as intended. For
convenience, we have a Makefile directive that runs all tests automatically.

```
make test
```

All tests must pass for a pull request to be accepted!

## Tutorials

All tutorials are located in the [spateo-tutorials](https://github.com/aristoteleo/spateo-tutorials) GitHub repo. We also provide a Jupyter Notebook template for tutorials [here](https://github.com/aristoteleo/spateo-tutorials/blob/main/template.ipynb).