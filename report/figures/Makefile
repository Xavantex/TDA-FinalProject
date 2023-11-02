# This Makefile is to be located in a subdir "figures", where all
# .fig-images are stored too.

SHELL := /bin/bash


FIGURES    := $(wildcard *.fig)
FIGPDFTEX  := $(patsubst %.fig,%.pdf,$(FIGURES))
FIGPDFTEXT := $(patsubst %.fig,%.pdftex_t,$(FIGURES))

%.pdf : %.fig
	fig2dev -L pdftex -n $@ $< $@

%.pdftex_t : %.fig
	# pdftex is pdf WITHOUT text in it.  But pdftex_t-file must
	# includegraphics{} a pdf file, so change name
	fig2dev -L pdftex_t -p $*.pdftex $< $@
	sed -i 's|includegraphics{$*.pdftex|includegraphics{figures/$*.pdf|' $@


.PHONY:	fig

fig:	$(FIGPDFTEXT) $(FIGPDFTEX)


clean:
	rm *.pdftex_t *pdf
