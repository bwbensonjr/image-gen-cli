.PHONY: help sync test build clean publish-test publish

help:
	@echo "Targets:"
	@echo "  sync          Install/refresh dependencies from uv.lock"
	@echo "  test          Run pytest suite"
	@echo "  build         Build sdist and wheel into dist/"
	@echo "  clean         Remove build artifacts"
	@echo "  publish-test  Upload dist/* to TestPyPI (requires UV_PUBLISH_TOKEN)"
	@echo "  publish       Upload dist/* to PyPI (requires UV_PUBLISH_TOKEN)"

sync:
	uv sync

test:
	uv run pytest

build: clean
	uv build

clean:
	rm -rf dist build *.egg-info

publish-test: build
	uv publish --publish-url https://test.pypi.org/legacy/

publish: build
	uv publish
