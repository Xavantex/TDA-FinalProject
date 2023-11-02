DOC=report

all:	${DOC}.pdf

#${DOC}.pdf:	${DOC}.tex figures plots references.bib
${DOC}.pdf:	${DOC}.tex figures references.bib
#	cd plots && make
	cd figures && make
	latexmk -pdf -pdflatex="pdflatex -interactive=nonstopmode -shell-escape" -use-make ${DOC}.tex

clean:
	rm -f ${DOC}.pdf

#plots:	FORCE
#	cd plots && make

figures:	FORCE
	cd figures && make

FORCE:
