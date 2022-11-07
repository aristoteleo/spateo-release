.PHONY : install install-dev install-all test check build docs clean push_release

install:
	pip install .
	# There is a problem with just pip installing hdbscan...
	# pip uninstall -y hdbscan
	# pip install --no-build-isolation --no-binary :all: hdbscan>=0.8.26

install-dev:
	pip install -r dev-requirements.txt

install-tdr:
	pip install -r 3d-requirements.txt

install-docs:
	pip install -r docs/requirements.txt

install-all: install-dev install-docs install

test:
	rm -f .coverage
	nosetests --verbose --with-coverage --cover-package spateo \
		tests/* \
		tests/io/* \
		tests/preprocessing/* \
		tests/segmentation/* \
		tests/tools/*

check:
	isort --profile black --check spateo tests && black --check spateo tests && echo OK

build:
	python setup.py sdist

docs:
	sphinx-build -a docs docs/_build

clean:
	rm -rf build
	rm -rf dist
	rm -rf spateo.egg-info
	rm -rf docs/_build
	rm -rf docs/autoapi
	rm -rf .coverage

bump_patch:
	bumpversion patch

bump_minor:
	bumpversion minor

bump_major:
	bumpversion major

push_release:
	git push && git push --tags
