.PHONY : install install-dev install-all test check build docs clean push_release

install:
	pip install --no-build-isolation --no-binary hdbscan -r requirements.txt

install-dev:
	pip install -r dev-requirements.txt

install-all: install-dev install

test:
	rm -f .coverage
	nosetests --verbose --with-coverage --cover-package spateo \
		tests/* \
		tests/io/* \
		tests/preprocessing/* \
		tests/preprocessing/segmentation/* \
		tests/tools/*

check:
	black spateo tests --check && echo OK

build:
	python setup.py sdist

docs:
	sphinx-build -a docs docs/_build

clean:
	rm -rf build
	rm -rf dist
	rm -rf spateo.egg-info
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf .coverage

bump_patch:
	bumpversion patch

bump_minor:
	bumpversion minor

bump_major:
	bumpversion major

push_release:
	git push && git push --tags
