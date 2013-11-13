# === func-gen- : muon_simulation/presentation/slides fgp muon_simulation/presentation/slides.bash fgn slides fgh muon_simulation/presentation
slides-src(){      echo muon_simulation/presentation/slides.bash ; }
slides-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slides-src)} ; }
slides-vi(){       vi $(slides-source) ; }
slides-env(){      elocal- ; }
slides-usage(){ cat << EOU

S5 SLIDES PDF 
================

Creates PDF documents from a sequence of cropped browser 
screen capture PNGs. 

This is particularly useful with S5 slides created with 
rst2s5.py as this avoids having to duplicate the layout exercise 
for the pdf.  A PDF can be created from the rst via several routes
but it will not look like the S5 slides without duplicated styling 
effort.

The disadvantage is bitmapped PDFs lacking clickable links. 
Mitigate this (and gain some html traffic) by providing a 
prominent reference to the online html version of the slides.

The advantage of having slides in html, generated from 
plain text RST files outweighs the disadvantage.  

MINIMAL SPHINX INTEGRATION
-----------------------------

Minimal integration with html docs created by Sphinx is 
achieved by:

#. plain docutils sources for rst2s5 (not sphinx) 
   are named *.txt* rather than *.rst* 

#. generated S5 html and pdf are placed within the Sphinx
   build directory at the appropriate place in the tree
   corresponding to the RST sources

#. the Sphinx index.rst contains a raw html list of links
   to the rst2s5 generated html and pdf 

FUNCTIONS
----------

*slides-get N M*
                performs the below functions, does screencaptures to PNG, 
                cropping PNG and converting to PDF, usage::

                     slides-get 0 5    


*slides-capture N M*
                screencapture a sequence of Safari pages, eg S5 slides

                During operation the sequence of Browser pages will load
                one by one.  As each URL is visited, user intervention 
                to click the window is required. As the tabs are left it is 
                preferable to start with only a small number of 
                Safari tabs before running the script.

                For each load:

                #. focus will shift to the new page
                #. wait until onload javascript has completed
                #. screencapture will kick in and color the browser blue with a camera icon
                #. click the browser window to take the capture

*slides-crop*
              runs python cropper on all NN.png, creating NN_crop.png
*slides-convert*
              uses convert to concatenate NN_crop.png into name.pdf


*slides-rst2pdf-convert*
              DID NOT PURSUE THIS TECHNIQUE AS TOO MUCH STYLING REINVENTION

TODO
------

#. recreate PDF once links/structure finalized

#. announce the link 


EOU
}

slides-fold(){  echo $(slides-branch)/$(slides-name) ; }
slides-dir(){   echo $(local-base)/env/$(slides-fold) ; }
slides-sdir(){  echo $(env-home)/$(slides-branch) ; } 
slides-pdir(){  echo $(env-home)/_build/dirhtml/$(slides-fold) ; }

slides-cd(){  cd $(slides-dir); }
slides-scd(){  cd $(slides-sdir); }
slides-mate(){ mate $(slides-dir) ; }
slides-mkdir(){ mkdir -p $(slides-dir) ; }
slides-get(){

   slides-mkdir
   slides-cd
   slides-capture $*
   slides-crop
   slides-convert

}

slides-name(){      echo ${SLIDES_NAME:-nov2013_gpu_nuwa} ; }
slides-branch(){    echo ${SLIDES_BRANCH:-muon_simulation/presentation} ; }        # env relative path to where .txt sources reside
slides-host(){      echo ${SLIDES_HOST:-dayabay.phys.ntu.edu.tw} ; }   
slides-urlbase(){   echo ${SLIDES_URLBASE:-http://$(slides-host)/e} ; }   
slides-url(){       echo ${SLIDES_URL:-http://$(slides-host)/e/$(slides-fold)/$(slides-name).html} ; }
slides-url-page(){  echo "$(slides-url)?p=$1" ; }

slides-pages(){
  local i
  local START=${1:-0}
  local END=${2:-0}
  typeset -i START END 
  for ((i=START;i<=END;++i)); do echo $i; done
}

slides-make(){
   local msg="=== $FUNCNAME "
   slides-scd

   echo $msg creating S5 html slides from txt source
   make

   local outdir=$(slides-dir)
   mkdir -p $outdir
   echo $msg rsync S5 html slides and sources to $outdir
   make rsync OUTDIR=$outdir
}

slides-publish(){
  case $NODE_TAG in 
     C2|C2R) $FUNCNAME-rsync ;;
          *) $FUNCNAME-ln ;;
  esac
}

slides-publish-ln(){
   local pdir=$(slides-pdir)
   mkdir -p $(dirname $pdir)
   ln -svf $(slides-dir)/ $pdir
   ls -l $(dirname $pdir)
}
slides-publish-rsync(){
   local pdir=$(slides-pdir)
   mkdir -p $(dirname $pdir)
   rsync -av $(slides-dir)/ $pdir/
}

slides-capture(){
   local msg="=== $FUNCNAME "
   slides-cd
   local pages=$(slides-pages $1 $2)
   local page
   local url
   local zpage
   for page in $pages ; do
      url=$(slides-url-page $page)
      zpage=$(printf "%0.2d" $page)
      echo $msg opening url "$url" 
      open "$url"
      cmd="screencapture -T0 -w -i -o $zpage.png"
      #
      #    -T<seconds>  Take the picture after a delay of <seconds>, default is 5 
      #    -w           only allow window selection mode
      #    -i           capture screen interactively, by selection or window
      #    -o           in window capture mode, do not capture the shadow of the window
      #
      echo $msg about to do $cmd : tap browser window once loaded and highlighted blue
      sleep 3
      eval $cmd
   done
}
slides-crop(){
   local msg="=== $FUNCNAME "
   slides-cd
   echo $msg cropping PNG 
   crop.py ??.png   
}
slides-convert(){
   local msg="=== $FUNCNAME "
   local pdf=$(slides-name).pdf   
   slides-cd
   echo $msg converting PNG into $pdf 
   convert ??_crop.png $pdf
}


slides-rst2pdf(){
   /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2pdf $* 
}
slides-rst2pdf-convert(){
  local name=$(slides-name)
  #slides-rst2pdf $name.txt -o $name.pdf
  slides-rst2pdf $name.txt -b1 -s slides.style -o $name.pdf
}


