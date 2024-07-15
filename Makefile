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
	@echo "➡️ Running Black..."
	python3 -m black --check .

	@echo "➡️ Running deptry..."
	python3 -m deptry \
		--ignore-notebooks \
		--per-rule-ignores "DEP002=aiohttp" \
		--per-rule-ignores "DEP003=aiohttp_retry" \
		.

lint:
	@echo "➡️ Running Black..."
	python3 -m black .

dev:
	VERSION=$(version_full) func start
