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



HMM local file preview has all image links broken

* file:///usr/local/simoncblyth.github.io/images/index.html

despite the ones beginning 


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

case $(uname) in 
 Darwin)  RST2HTML=rst2html-3.13 ; OPEN=open ;;
 Linux)   RST2HTML=rst2html      ; OPEN="gio open" ;;
esac

GITHUB_HTDOCS=/usr/local/simoncblyth.github.io/images

if [ ! -d "$GITHUB_HTDOCS" ]; then
   echo $BASH_SOURCE - ERROR GITHUB_HTDOCS $GITHUB_HTDOCS
   exit 1
fi

cp custom.css $GITHUB_HTDOCS/custom.css

$RST2HTML --stylesheet=$GITHUB_HTDOCS/custom.css index.txt $GITHUB_HTDOCS/index.html

$OPEN $GITHUB_HTDOCS/index.html


