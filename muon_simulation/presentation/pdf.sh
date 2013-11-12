#!/bin/bash -l 

#rst2pdf(){
#   /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2pdf $* 
#}
#
#NAME=nov2013_gpu_nuwa
#rst2pdf $NAME.txt -o $NAME.pdf
#
#rst2pdf $NAME.txt -b1 -s slides.style -o $NAME.pdf


slide-usage(){ cat << EOU

SCREENCAPTURE A SEQUENCE OF SAFARI PAGES (FOR EXAMPLE S5 SLIDES)
===================================================================

Usage::

   ./pdf.sh

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


CROPPING SAFARI CHROME
------------------------

::

    simon:nov2013 blyth$ crop.py ??.png
    INFO:env.doc.crop:Crop safari_headtail vertically chop the head by param[0] and tail by param[1] 
    INFO:env.doc.crop:cropping 00.png to create 00_crop.png 
    INFO:env.doc.crop:cropping 01.png to create 01_crop.png 
    INFO:env.doc.crop:cropping 02.png to create 02_crop.png 
    INFO:env.doc.crop:cropping 03.png to create 03_crop.png 
    INFO:env.doc.crop:cropping 04.png to create 04_crop.png 
    INFO:env.doc.crop:cropping 05.png to create 05_crop.png 

Concatenate the PNG to make a PDF::

    convert 0?_crop.png nov2013_gpu_nuwa.pdf


EOU
}

slide-url-base(){ echo http://belle7.nuu.edu.tw/muon_simulation/nov2013/nov2013_gpu_nuwa.html ; }
slide-url(){      echo "$(slide-url-base)?p=$1" ; }
slide-pages(){
  local i
  local START=$1
  local END=$2
  typeset -i START END 
  for ((i=START;i<=END;++i)); do echo $i; done
}

slide-capture(){
   local msg="=== $FUNCNAME "
   local pages=$(slide-pages $1 $2)
   local page
   local url
   local zpage
   for page in $pages ; do
      url=$(slide-url $page)
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

slide-capture 6 17



