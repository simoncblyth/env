#!/usr/bin/env bash
usage(){ cat << EOU
~/env/simoncblyth.github.io/images/make.sh
============================================

Update the index page::

   https://simoncblyth.github.io/images/

Workflow edit source in env repo::

   cd ~/env/simoncblyth.github.io/images
   vi index.txt 

Convert that to html with rst2html (needs miniconda env with docutils)::

   ~/env/simoncblyth.github.io/images/make.sh

EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

#RST2HTML=rst2html-3.13
RST2HTML=rst2html
GITHUB_HTDOCS=/usr/local/simoncblyth.github.io/images

if [ ! -d "$GITHUB_HTDOCS" ]; then
   echo $BASH_SOURCE - ERROR GITHUB_HTDOCS $GITHUB_HTDOCS
   exit 1
fi

cp custom.css $GITHUB_HTDOCS/custom.css

$RST2HTML --stylesheet=$GITHUB_HTDOCS/custom.css index.txt $GITHUB_HTDOCS/index.html

gio open $GITHUB_HTDOCS/index.html

