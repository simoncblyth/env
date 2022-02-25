#
# Convert RST images/index.txt into images/index.html within $(BITBUCKET_HTDOCS)/images 
# to go live at http://simoncblyth.bitbucket.io/images
#
# See bitbucketstatic-vi for how this is used
#

IMAGES_HTDOCS=$(HOME)/simoncblyth.bitbucket.io/images

.PHONY: all 

all: $(IMAGES_HTDOCS)/custom.css $(IMAGES_HTDOCS)/index.html
	@echo thats $@ folks : $< 
	@open $(IMAGES_HTDOCS)/index.html

$(IMAGES_HTDOCS)/custom.css : custom.css
	mkdir -p $(IMAGES_HTDOCS) && cp $< $@

$(IMAGES_HTDOCS)/index.html : index.txt custom.css
	mkdir -p $(IMAGES_HTDOCS) && rst2html-2.7.py --stylesheet=$(IMAGES_HTDOCS)/custom.css $< $@ 

