# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS      ?=
SPHINXBUILD     ?= sphinx-build
SPHINXAPIDOC    ?= sphinx-apidoc  # Define the apidoc command
SOURCEDIR       = source
BUILDDIR        = build
PACKAGE_PATH    = ../src/cleave_app # Path to your Python package

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc

# NEW RULE: Define how to run sphinx-apidoc correctly
apidoc:
	@echo "Running sphinx-apidoc..."
	# Clean the old api directory first
	@rm -rf $(SOURCEDIR)/api
	# Run the correct command with the -M flag
	@$(SPHINXAPIDOC) -f -e -M -o $(SOURCEDIR)/api $(PACKAGE_PATH)

# MODIFIED RULE: Make `html` depend on `apidoc`
html: apidoc
	@echo "Building HTML..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# MODIFIED RULE: Make `clean` also clean the api directory
clean:
	@echo "Cleaning build and api directories..."
	@rm -rf $(BUILDDIR)/*
	@rm -rf $(SOURCEDIR)/api

# Catch-all target: route all other unknown targets to Sphinx
# We keep this for commands like `make latexpdf`, etc.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)