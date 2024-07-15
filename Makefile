# Versioning
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)

version:
	@bash ./cicd/version/version.sh -g . -c

version-full:
	@bash ./cicd/version/version.sh -g . -c -m

install:
	@echo "➡️ Installing pip-tools..."
	python3 -m pip install pip-tools

	@echo "➡️ Syncing dependencies..."
	pip-sync --pip-args "--no-deps" requirements-dev.txt

upgrade:
	@echo "➡️ Upgrading pip..."
	python3 -m pip install --upgrade pip setuptools wheel

	@echo "➡️ Upgrading pip-tools..."
	python3 -m pip install --upgrade pip-tools

	@echo "➡️ Compiling app requirements..."
	pip-compile \
		--output-file requirements.txt \
		pyproject.toml

	@echo "➡️ Compiling dev requirements..."
	pip-compile \
		--extra dev \
		--output-file requirements-dev.txt \
		pyproject.toml

test:
	@echo "➡️ Test generic formatter (Black)..."
	python3 -m black --check .

	@echo "➡️ Test import formatter (isort)..."
	python3 -m isort --jobs -1 --check .

	@echo "➡️ Test dependencies issues (deptry)..."
	python3 -m deptry .

	@echo "➡️ Test code smells (Pylint)..."
	python3 -m pylint .

	@echo "➡️ Test types (Pyright)..."
	python3 -m pyright .

lint:
	@echo "➡️ Fix with generic formatter (Black)..."
	python3 -m black .

	@echo "➡️ Fix with import formatter (isort)..."
	python3 -m isort --jobs -1 .

dev:
	VERSION=$(version_full) func start
