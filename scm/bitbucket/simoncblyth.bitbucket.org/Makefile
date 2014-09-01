#
# Convert RST index.txt into index.html within $(BITBUCKET_HTDOCS) 
# to go live at http://simoncblyth.bitbucket.org
#
# See bitbucket-vi for how this is used
#

BITBUCKET_HTDOCS=$(HOME)/simoncblyth.bitbucket.org

$(BITBUCKET_HTDOCS)/index.html : index.txt
	mkdir -p $(BITBUCKET_HTDOCS) && rst2html-2.7.py $< $@ 

