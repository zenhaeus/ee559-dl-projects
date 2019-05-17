FILENAME=report
REFERENCES=literature

.SUFFIXES:
.SUFFIXES: .bib .pdf .tex
.PHONY: clean

run: $(FILENAME).pdf

$(FILENAME).pdf: $(FILENAME).bbl $(FILENAME).tex
	pdflatex $(FILENAME).tex -draftmode
	pdflatex $(FILENAME).tex 

clean:
	rm -rf *.aux *.lof *.log *.lot *.toc *.bbl *.blg *pdf
