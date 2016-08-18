# Usage:
#
# * presentation-vi to set the name of the presentation
# * presentation-make to run this make file with the needed environment   
#
#
# Convert RST .txt into .html within $(DOCBASE) and 
# open a page of the presentation in Safari:
# 
#    make open PAGE=6
# 
#    make touch ; make open PAGE=6    # if modifying smth not detected by this Makefile
#
#
PYTHON?=/opt/local/bin/python

#FOOTNOTE_REFERENCES="superscript"
FOOTNOTE_REFERENCES="brackets"

PAGE=0
REMOTEHOST=simoncblyth.bitbucket.org
DOCBASE=$(HOME)/simoncblyth.bitbucket.org

SOURCES=$(PRESENTATION_NAME).txt

REALPWD=$(realpath $(PWD))   # avoids requirement to be at ~/env/wherever rather than symbolically linked ~/e/wherever
RELDIR=$(subst $(ENV_HOME),,$(REALPWD)) 
PUBDIR=$(addprefix $(DOCBASE)/env,$(RELDIR))
PUBURL=$(addprefix http://localhost/env,$(RELDIR))
PUBURLR=$(addprefix http://$(REMOTEHOST)/env,$(RELDIR))

SLIDES_HTML=$(subst .txt,.html,$(SOURCES))

SLIDES=$(addprefix $(PUBDIR)/,$(SLIDES_HTML))
PDFS=$(addprefix $(PUBDIR)/,$(subst .txt,.pdf,$(SOURCES)))
TXTS=$(addprefix $(PUBDIR)/,$(SOURCES))
SLIDES_URL=$(addsuffix ?page=$(PAGE),$(addprefix $(PUBURL)/,$(SLIDES_HTML)))
SLIDES_RURL=$(addsuffix ?page=$(PAGE),$(addprefix $(PUBURLR)/,$(SLIDES_HTML)))


.PHONY: clean slides touch info ls open help

slides: $(SLIDES) $(TXTS)
	@echo created $^
	@echo local/remote urls : remote needs hg commit/push to update
	@echo open $(SLIDES_URL)
	@echo open $(SLIDES_RURL) 

$(TXTS): $(SOURCES)
	@cp $< $@

$(SLIDES): $(SOURCES)
	mkdir -p $(PUBDIR) && DOCBASE=$(DOCBASE) $(PYTHON) ./rst2s5-2.6.py --traceback --footnote-references=$(FOOTNOTE_REFERENCES) --theme-url ui/my-small-white --current-slide --visible-controls --language=en $< $@




open: $(SLIDES)
	@echo open $(SLIDES_URL)

info:
	@echo RELDIR $(RELDIR)
	@echo PUBDIR $(PUBDIR)
	@echo PUBURL $(PUBURL)
	@echo SLIDES $(SLIDES)
	@echo SLIDES_HTML $(SLIDES_HTML)
	@echo SLIDES_URL $(SLIDES_URL)
	@echo PDFS $(PDFS)
	@echo TXTS $(TXTS)

ls:
	@echo PUBDIR $(PUBDIR)
	@ls -l $(PUBDIR)

touch: 
	touch $(SOURCES)

help:
	$(PYTHON) ./rst2s5-2.6.py --help

clean:
	rm -f $(SLIDES) $(PDFS)



