#
# Convert RST .txt into .html within $(BITBUCKET_HTDOCS) 
# to go live at 
#
#
# Steps to update index page http://simoncblyth.bitbucket.org
#
# :: 
#
#     bitbucket-
#
#     cd ~/env/scm/bitbucket/simoncblyth.bitbucket.org    # bitbucket-scd
#
#     make       ## writes ~/simoncblyth.bitbucket.org/index.html
#
#     cd ~/simoncblyth.bitbucket.org    # bitbucket-cd
#
#     hg commit -m "update top index "
#
#     hg push
#
#     open http://simoncblyth.bitbucket.org
#
#

BITBUCKET_HTDOCS=$(HOME)/simoncblyth.bitbucket.org

$(BITBUCKET_HTDOCS)/index.html : index.txt
	mkdir -p $(BITBUCKET_HTDOCS) && rst2html-2.7.py $< $@ 

