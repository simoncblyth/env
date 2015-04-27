#
# Convert RST index.txt into index.html within $(BITBUCKET_HTDOCS) 
# to go live at http://simoncblyth.bitbucket.org
#
# See bitbucketstatic-vi for how this is used
#

BITBUCKET_HTDOCS=$(HOME)/simoncblyth.bitbucket.org

.PHONY: all 

all: $(BITBUCKET_HTDOCS)/custom.css $(BITBUCKET_HTDOCS)/index.html
	@echo thats $@ folks : $< 

$(BITBUCKET_HTDOCS)/custom.css : custom.css
	mkdir -p $(BITBUCKET_HTDOCS) && cp $< $@

$(BITBUCKET_HTDOCS)/index.html : index.txt custom.css
	mkdir -p $(BITBUCKET_HTDOCS) && rst2html-2.7.py --stylesheet=$(BITBUCKET_HTDOCS)/custom.css $< $@ 

