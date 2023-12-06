#!/bin/bash

rm -fv impact.blg impact.aux impact.out impact.log impact.pdf impact_grayscale.pdf

pdflatex impact.tex
bibtex impact
pdflatex impact.tex
pdflatex impact.tex
