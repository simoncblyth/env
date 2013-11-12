PYTHON?=python

SOURCES=$(wildcard *.txt)
SLIDES=$(subst .txt,.html,$(SOURCES))
OUTDIR=$(ENV_HOME)/_build/dirhtml/muon_simulation/presentation

.PHONY: sync clean

slides: $(SLIDES)

%.html: %.txt
	$(PYTHON) ./rst2s5-2.6.py --theme-url ui/my-small-white --current-slide --visible-controls --language=en $< $@


sync:
	@echo copying SLIDES $(SLIDES) SOURCES $(SOURCES) to OUTDIR $(OUTDIR)
	cp $(SLIDES) $(SOURCES) $(OUTDIR)/
	cp -R ui images $(OUTDIR)/
    

clean:
	rm -f $(SLIDES)
