PYTHON?=python

SOURCES=$(wildcard *.txt)
SLIDES=$(subst .txt,.html,$(SOURCES))
PDFS=$(subst .txt,.pdf,$(SOURCES))

OUTDIR=$(ENV_HOME)/_build/dirhtml/muon_simulation/presentation

.PHONY: rsync clean slides touch

slides: $(SLIDES)
	@echo created $<

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
