#!/bin/bash

# This script is for bash shells like MINGW64 or Git Bash.

echo "=========================================================="
echo " CLEANING OLD DOCUMENTATION"
echo "=========================================================="
# Use 'rm -rf' which is the bash command for deleting directories
rm -rf build
rm -rf source/api

echo ""
echo "=========================================================="
echo " GENERATING API SOURCE FILES (.rst)"
echo "=========================================================="
# Use forward slashes for paths in bash
sphinx-apidoc -f -e -M -o ./docs/source/api ./src/cleave_app 

echo ""
echo "=========================================================="
echo " [3/4] FORCIBLY DELETING UNWANTED .rst FILES"
echo "=========================================================="
# This is the step that fixes your problem. We delete the files by name.
echo "Deleting source/api/cleave_app.rst..."
rm -f ./docs/source/api/cleave_app.rst || true

echo "Deleting source/api/modules.rst..."
rm -f ./docs/source/api/modules.rst || true

echo ""
echo "=========================================================="
echo " BUILDING HTML"
echo "=========================================================="
sphinx-build -b html docs/source build

echo ""
echo "=========================================================="
echo " Build finished. Open 'build/index.html' in your browser."
echo "=========================================================="