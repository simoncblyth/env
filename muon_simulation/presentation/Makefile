PYTHON?=python

SOURCES=$(wildcard *.txt)
SLIDES=$(subst .txt,.html,$(SOURCES))
OUTDIR=$(ENV_HOME)/_build/dirhtml/muon_simulation/presentation

.PHONY: rsync clean

slides: $(SLIDES)

%.html: %.txt
	$(PYTHON) ./rst2s5-2.6.py --theme-url ui/my-small-white --current-slide --visible-controls --language=en $< $@

rsync:
	rsync -av $(SLIDES) $(OUTDIR)/
	rsync -av $(SOURCES) $(OUTDIR)/
	rsync -av ui $(OUTDIR)/
	rsync -av images $(OUTDIR)/

clean:
	rm -f $(SLIDES)
