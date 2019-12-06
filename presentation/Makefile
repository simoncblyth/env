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

REMOTEHOST=simoncblyth.bitbucket.io
DOCBASE=$(HOME)/simoncblyth.bitbucket.io

PREPROCESSOR_ARGS=$(PRESENTATION_PREPROCESSOR_ARGS)

REALPWD=$(realpath $(PWD))   # avoids requirement to be at ~/env/wherever rather than symbolically linked ~/e/wherever
RELDIR=$(subst $(ENV_HOME),,$(REALPWD)) 

PUBDIR=$(addprefix $(DOCBASE)/env,$(RELDIR))
PUBURL=$(addprefix http://localhost/env,$(RELDIR))
PUBURLR=$(addprefix http://$(REMOTEHOST)/env,$(RELDIR))


INAME=$(PRESENTATION_INAME)
ONAME=$(PRESENTATION_ONAME)

ITXT=$(INAME).txt
IDEF=my_s5defs.txt
IIMG=s5_background_image.txt


OHTML=$(addprefix $(PUBDIR)/,$(ONAME).html)
OPDF=$(addprefix $(PUBDIR)/,$(ONAME).pdf))
OTXT=$(addprefix $(PUBDIR)/,$(ONAME).txt)
ODEF=$(addprefix $(PUBDIR)/,$(IDEF))
OIMG=$(addprefix $(PUBDIR)/,$(IIMG))

SLIDES_URL=$(addsuffix ?page=$(PAGE),$(addprefix $(PUBURL)/,$(ONAME).html))
SLIDES_RURL=$(addsuffix ?page=$(PAGE),$(addprefix $(PUBURLR)/,$(ONAME).html))


.PHONY: clean slides touch info ls open help

slides: $(OHTML) $(OTXT)
	@echo Makefile.created $^
	@echo local/remote urls : remote needs hg commit/push to update
	@echo open $(SLIDES_URL)
	@echo open $(SLIDES_RURL) 

$(ODEF): $(IDEF)
	@echo Makefile.cp definitions into place
	@cp $< $@

$(OIMG): $(IIMG)
	@echo Makefile.cp backgound img refs into place
	@cp $< $@




$(OTXT): $(ITXT)
	@echo Makefile.gcc preprocess source RST picking ifdef 
	@gcc -E -P -traditional-cpp -x c $(PREPROCESSOR_ARGS) $< > $@

$(OHTML): $(OTXT) $(ODEF) $(OIMG)
	@echo Makefile.rst2s5
	@echo ITXT $(ITXT)
	@echo OTXT $(OTXT)
	@echo ODEF $(ODEF)
	@echo OHTML $(OHTML)  --visible-controls
	mkdir -p $(PUBDIR) && DOCBASE=$(DOCBASE) $(PYTHON) ./rst2s5-2.6.py --traceback --footnote-references=$(FOOTNOTE_REFERENCES) --theme-url ui/my-small-white --current-slide --language=en $< $@

open: $(OHTML)
	@echo Makefile.open
	@echo open $(SLIDES_URL)

info:
	@echo Makefile.info
	@echo RELDIR $(RELDIR)
	@echo PUBDIR $(PUBDIR)
	@echo PUBURL $(PUBURL)
	@echo OHTML $(OHTML)
	@echo SLIDES_URL $(SLIDES_URL)
	@echo OPDF $(OPDF)
	@echo OTXT $(OTXT)

ls:
	@echo PUBDIR $(PUBDIR)
	@ls -l $(PUBDIR)

touch: 
	touch $(ITXT)

help:
	$(PYTHON) ./rst2s5-2.6.py --help

clean:
	rm -f $(OHTML) $(OPDF)



