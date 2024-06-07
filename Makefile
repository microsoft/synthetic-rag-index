# Versioning
version_full ?= $(shell $(MAKE) --silent version-full)
version_small ?= $(shell $(MAKE) --silent version)

version:
	@bash ./cicd/version/version.sh -g . -c

version-full:
	@bash ./cicd/version/version.sh -g . -c -m

install:
	@for f in $$(find . -name "requirements*.txt"); do \
		echo "➡️ Installing Python dependencies in $$f..."; \
		python3 -m pip install -r $$f; \
	done

upgrade:
	@echo "➡️ Upgrading pip..."
	python3 -m pip install --upgrade pip

	@for f in $$(find . -name "requirements*.txt"); do \
		echo "➡️ Upgrading Python dependencies in $$f..."; \
		python3 -m pip install --upgrade -r $$f; \
	done

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
