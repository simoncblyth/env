#
# Convert RST .txt into .html within $(APACHE_HTDOCS) and 
# open a page of the presentation in Safari:
# 
#    make open PAGE=6
# 
#    make touch ; make open PAGE=6    # if modifying smth not detected by this Makefile
#
#
PYTHON?=python

RELDIR=$(subst $(ENV_HOME),,$(PWD))   # hmm relies on being at ~/env/wherever rather than symbolically linked ~/e/wherever
NEWPUBDIR=$(addprefix $(APACHE_HTDOCS)/env,$(RELDIR))
NEWPUBURL=$(addprefix http://localhost/env,$(RELDIR))

SOURCES=$(wildcard *.txt)
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


.PHONY: rsync clean slides touch info ls open

slides: $(SLIDES) $(TXTS)
	@echo created $< NAMES $(NAMES)

info:
	@echo RELDIR $(RELDIR)
	@echo NEWPUBDIR $(NEWPUBDIR)
	@echo SLIDES $(SLIDES)
	@echo SLIDES_URL $(SLIDES_URL)
	@echo TXTS $(TXTS)

ls:
	@echo NEWPUBDIR $(NEWPUBDIR)
	@ls -l $(NEWPUBDIR)

open: $(SLIDES)
	@open $(SLIDES_URL)

touch: 
	touch $(SOURCES)

$(TXTS): $(SOURCES)
	cp $< $@

$(SLIDES): $(SOURCES)
	$(PYTHON) ./rst2s5-2.6.py --traceback --theme-url ui/my-small-white --current-slide --visible-controls --language=en $< $@

rsync:
	rsync -av $(SLIDES) $(OUTDIR)/
	rsync -av $(SOURCES) $(OUTDIR)/
	rsync -av ui $(OUTDIR)/
	rsync -av images $(OUTDIR)/
	#rsync -av $(PDFS) $(OUTDIR)/

clean:
	rm -f $(SLIDES) $(PDFS)
