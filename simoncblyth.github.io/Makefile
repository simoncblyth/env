#
# Convert RST index.txt into index.html within $(GITHUB_HTDOCS) 
# to go live at http://simoncblyth.bitbucket.io
#
# See bitbucketstatic-vi for how this is used
#

GITHUB_HTDOCS=$(HOME)/simoncblyth.github.io

.PHONY: all 

all: $(GITHUB_HTDOCS)/custom.css $(GITHUB_HTDOCS)/index.html
	@echo thats $@ folks : $< 
	@open $(GITHUB_HTDOCS)/index.html

$(GITHUB_HTDOCS)/custom.css : custom.css
	mkdir -p $(GITHUB_HTDOCS) && cp $< $@

$(GITHUB_HTDOCS)/index.html : index.txt custom.css
	mkdir -p $(GITHUB_HTDOCS) && rst2html-3.13 --stylesheet=$(GITHUB_HTDOCS)/custom.css $< $@ 

