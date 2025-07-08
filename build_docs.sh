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
sphinx-apidoc -f -e -M -o source/api ../src/cleave_app 

echo ""
echo "=========================================================="
echo " FORCIBLY DELETING UNWANTED .rst FILES"
echo "=========================================================="

echo "Deleting source/api/modules.rst..."
#rm -f source/api/modules.rst || true

echo ""
echo "=========================================================="
echo " BUILDING HTML"
echo "=========================================================="
sphinx-build -b html source build

echo ""
echo "=========================================================="
echo " Build finished. Open 'build/index.html' in your browser."
echo "=========================================================="