PYTHON?=python

SOURCES=$(wildcard *.txt)
NAMES=$(subst .txt,,$(SOURCES))
SLIDES=$(subst .txt,.html,$(SOURCES))
PDFS=$(subst .txt,.pdf,$(SOURCES))

RELDIR=$(subst $(ENV_HOME),,$(PWD))   # hmm relies on being at ~/env/wherever rather than symbolically linked ~/e/wherever

PUBDIR=$(ENV_HOME)/_build/dirhtml/muon_simulation/presentation
OUTDIR=$(LOCAL_BASE)/env/muon_simulation/presentation/name-of-presentation   # this is set exterally in slides-make

.PHONY: rsync clean slides touch info

slides: $(SLIDES)
	@echo created $< NAMES $(NAMES)

info:
	@echo RELDIR $(RELDIR)

touch: 
	touch $(SOURCES)

%.html: %.txt
	$(PYTHON) ./rst2s5-2.6.py --theme-url ui/my-small-white --current-slide --visible-controls --language=en $< $@

rsync:
	rsync -av $(SLIDES) $(OUTDIR)/
	rsync -av $(SOURCES) $(OUTDIR)/
	rsync -av ui $(OUTDIR)/
	rsync -av images $(OUTDIR)/
	#rsync -av $(PDFS) $(OUTDIR)/

clean:
	rm -f $(SLIDES) $(PDFS)
