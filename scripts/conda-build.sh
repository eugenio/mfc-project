#!/bin/bash
# Build script for MiKTeX to conda packages

conda-build .conda/miktex-recipe --output-folder .conda/packages