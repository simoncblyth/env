#
# Convert RST .txt into .html within $(APACHE_HTDOCS) and 
# open a page of the presentation in Safari:
# 
#    make open PAGE=6
# 
#    make touch ; make open PAGE=6    # if modifying smth not detected by this Makefile
#
#

#
PYTHON?=python

#FOOTNOTE_REFERENCES="superscript"
FOOTNOTE_REFERENCES="brackets"
APACHE_HTDOCS=$(HOME)/simoncblyth.bitbucket.org

REALPWD=$(realpath $(PWD))   # avoids requirement to be at ~/env/wherever rather than symbolically linked ~/e/wherever
RELDIR=$(subst $(ENV_HOME),,$(REALPWD)) 
NEWPUBDIR=$(addprefix $(APACHE_HTDOCS)/env,$(RELDIR))
#NEWPUBDIR=.
NEWPUBURL=$(addprefix http://localhost/env,$(RELDIR))

#SOURCES=$(wildcard *.txt)
#SOURCES=gpu_optical_photon_simulation.txt
SOURCES=g4dae_geometry_exporter.txt


NAMES=$(subst .txt,,$(SOURCES))
SLIDES_HTML=$(subst .txt,.html,$(SOURCES))
SLIDES=$(addprefix $(NEWPUBDIR)/,$(SLIDES_HTML))
PDFS=$(addprefix $(NEWPUBDIR)/,$(subst .txt,.pdf,$(SOURCES)))
TXTS=$(addprefix $(NEWPUBDIR)/,$(SOURCES))

SLIDES_URL=$(addsuffix ?page=$(PAGE),$(addprefix $(NEWPUBURL)/,$(SLIDES_HTML)))


# parasitic "integration" with sphinx is to be eliminated, as causes too many order sensitive steps to publish
PUBDIR=$(ENV_HOME)/_build/dirhtml/muon_simulation/presentation
# intermediary directory, also to be eliminated : go direct to APACHE_HTDOCS
OUTDIR=$(LOCAL_BASE)/env/muon_simulation/presentation/name-of-presentation   # this is set exterally in slides-make, used for rsync step only


.PHONY: rsync clean slides touch info ls open help

slides: $(SLIDES) $(TXTS)
	@echo created $< NAMES $(NAMES)

open: $(SLIDES)
	@echo open $(SLIDES_URL)

info:
	@echo RELDIR $(RELDIR)
	@echo NEWPUBDIR $(NEWPUBDIR)
	@echo SLIDES $(SLIDES)
	@echo SLIDES_URL $(SLIDES_URL)
	@echo TXTS $(TXTS)

ls:
	@echo NEWPUBDIR $(NEWPUBDIR)
	@ls -l $(NEWPUBDIR)

touch: 
	touch $(SOURCES)

$(TXTS): $(SOURCES)
	cp $< $@

$(SLIDES): $(SOURCES)
	mkdir -p $(APACHE_HTDOCS)/env/muon_simulation/presentation && DOCBASE=$(APACHE_HTDOCS) $(PYTHON) ./rst2s5-2.6.py --traceback --footnote-references=$(FOOTNOTE_REFERENCES) --theme-url ui/my-small-white --current-slide --visible-controls --language=en $< $@

help:
	$(PYTHON) ./rst2s5-2.6.py --help

rsync:
	rsync -av $(SLIDES) $(OUTDIR)/
	rsync -av $(SOURCES) $(OUTDIR)/
	rsync -av ui $(OUTDIR)/
	rsync -av images $(OUTDIR)/
	#rsync -av $(PDFS) $(OUTDIR)/

clean:
	rm -f $(SLIDES) $(PDFS)
