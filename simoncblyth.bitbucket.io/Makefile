#
# Convert RST index.txt into index.html within $(BITBUCKET_HTDOCS) 
# to go live at http://simoncblyth.bitbucket.io
#
# See bitbucketstatic-vi for how this is used
#

BITBUCKET_HTDOCS=$(HOME)/simoncblyth.bitbucket.io

.PHONY: all 

all: $(BITBUCKET_HTDOCS)/custom.css $(BITBUCKET_HTDOCS)/index.html
	@echo thats $@ folks : $< 
	@open $(BITBUCKET_HTDOCS)/index.html

$(BITBUCKET_HTDOCS)/custom.css : custom.css
	mkdir -p $(BITBUCKET_HTDOCS) && cp $< $@

$(BITBUCKET_HTDOCS)/index.html : index.txt custom.css
	mkdir -p $(BITBUCKET_HTDOCS) && rst2html-2.7.py --stylesheet=$(BITBUCKET_HTDOCS)/custom.css $< $@ 

