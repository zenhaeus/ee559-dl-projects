#!/usr/bin/env bash

# Compile report 1
cd ./Proj1/report
make
cd ../../

# Compile report 1
cd ./Proj1/report
make
cd ../../

zip -r Proj_285467_235440_226652.zip Proj1 Proj2 \
    --exclude Proj1/data\* \
    **/__pycache__\* \
    **/.ipynb_checkpoints\* \
    \*.aux \
    \*.blg \
    \*.ipynb \
    \*.out \
    \*.bib \
    \*.cls \
    \*.sty \
    \*.bst \
    \*.log \
    \*.bbl \
    **/report/Makefile \
    **/report/.gitignore \
    Proj1/output\*
