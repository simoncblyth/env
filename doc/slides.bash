# === func-gen- : muon_simulation/presentation/slides fgp muon_simulation/presentation/slides.bash fgn slides fgh muon_simulation/presentation
slides-src(){      echo doc/slides.bash ; }
slides-source(){   echo ${BASH_SOURCE:-$(env-home)/$(slides-src)} ; }
slides-vi(){       vi $(slides-source) ; }
slides-usage(){ cat << EOU

CONVERT SLIDES IN S5 RST TO HTML AND PDF 
=========================================

PDF page size : Very large because somehow 72dpi ?
---------------------------------------------------

* 2560x1440 points
* 90.32 x 50.8 cm


minor issue : some fullscreen image pages have a band at foot 860-846 of about 14 pixels
-------------------------------------------------------------------------------------------

eg /env/graphics/ggeoview/jpmt-inside-wide_crop.png 

Original image is : 2560x1440 


Fullscreen spec in s5_background_image is half that size::

   /env/graphics/ggeoview/jpmt-inside-wide_crop.png 1280px_720px

   Measuring safari window 

   83:722



aspect check
--------------

* slides safari sets aspect to 16/9
* use cross-hairs (shift-cmd-4) to check 


::

    tell application "Safari"

        set width to  1280
        set height to  720


::

    In [8]: 1280./720.
    Out[8]: 1.7777777777777777

    In [1]: 2560./1440.
    Out[1]: 1.7777777777777777

    In [3]: 16./9.
    Out[3]: 1.7777777777777777




g4dae
------

::

    delta:g4dae_geometry_exporter blyth$ pwd
    /Library/WebServer/Documents/env/presentation/g4dae_geometry_exporter


    delta:presentation blyth$ hg add 
    adding g4dae_geometry_exporter_okinawa.pdf
    env/presentation/g4dae_geometry_exporter_okinawa.pdf: up to 85 MB of RAM may be required to manage this file
    (use 'hg revert env/presentation/g4dae_geometry_exporter_okinawa.pdf' to cancel the pending addition)
    delta:presentation blyth$ 



Checking configured settings
-------------------------------

*slides-info*
    echos the paths/names in use

*slides-get 0 15*
    capture PNGs per page, crop the chrome, convert PNGs into PDF
    BEFORE USING MOVE ASIDE THE PRIOR CAPTURES MANUALLY::

        delta:~ blyth$ slides-cd ..
        delta:presentation blyth$ mv gpu_optical_photon_simulation gpu_optical_photon_simulation_apr28

*slides-get 0 0*   
    first page only 


Creating HTML slides from S5 Restructured Text Sources
--------------------------------------------------------

For simplicity arrange to have only one .txt file in the presentations folder, eg when 
adapting an existing presentation::

    simon:presentation blyth$ svn cp nov2013_gpu_nuwa.txt gpu_optical_photon_simulation.txt     
    A         gpu_optical_photon_simulation.txt
    simon:presentation blyth$ svn mv  nov2013_gpu_nuwa.txt  nov2013_gpu_nuwa.txt.old
    A         nov2013_gpu_nuwa.txt.old
    D         nov2013_gpu_nuwa.txt


Check/update *slides-name*, *slides-branch*, *slides-host* settings using *slides-vi* *slides-info*, to 
correspond to the name and location of the S5 sources. Normally 
would just need to update *slides-name*, in above example *gpu_optical_photon_simulation*.

Update S5 source, regenerate html and view local html in browser::

    simon:presentation blyth$ vi gpu_optical_photon_simulation.txt 
    simon:presentation blyth$ make
    python ./rst2s5-2.6.py --theme-url ui/my-small-white --current-slide --visible-controls --language=en gpu_optical_photon_simulation.txt gpu_optical_photon_simulation.html
    created gpu_optical_photon_simulation.html NAMES gpu_optical_photon_simulation

    simon:presentation blyth$ open gpu_optical_photon_simulation.html

To check a particular page use links like:

* file:///Users/blyth/env/muon_simulation/presentation/gpu_optical_photon_simulation.html?p=18


The above steps can also be done with *slides-make*.

*slides-make*
            Uses rst2s5.py to convert S5 .txt sources into .html.  


Rsync derived files elsewhere and cleanup source tree
-------------------------------------------------------

Check the slides dir destination is as desired then *make rsync*::

    simon:presentation blyth$ slides-dir
    /usr/local/env/muon_simulation/presentation/nov2013_gpu_nuwa
    simon:presentation blyth$ slides-
    simon:presentation blyth$ slides-dir
    /usr/local/env/muon_simulation/presentation/gpu_optical_photon_simulation


The above steps can also be done with *slides-rsync*.


*slides-rsync*
              Copies derived html and pdf files out of working copy, then cleans them from working copy.


Publish html to web server
----------------------------

#. make sure working copy is clean and commit changed files to subversion
#. update env working copy on webserver 
#. generate the html on the web server with *slides-make* or manually as shown above
#. use *slides-rsync* to propagate into the appropriate place for Sphinx derived html
#. rsync to the webserver htdocs with *slides-publish* note that the destination
   reuses the folders used by sphinx derived documentation

Hmm, also needed to::

   cd ~/env
   make rsync 

* TODO: streamline this, too many steps that have to be done in the correct order, see **eup** func on C2R


Publish to local apache
------------------------------

Avoid integration with Sphinx build docs which causes the complications.
Publish to local apache using:

#. *slides-apache-prepare*
#. *slides-apache-publish*


Publish to remote apache
---------------------------

#. *env-htdocs-rsync C2*



Integrate URLs of presentation HTML with Sphinx
--------------------------------------------------

In the Sphinx index.rst in the presention folder add 
raw html lists of links in order to allow navigation
from Sphinx derived docs to the presentation html and pdf.

Coexisting with Sphinx
-----------------------

Slide sources are named *.txt* rather than *.rst* 

This avoid Sphinx attempting to build the S5 sources, which are 
in plain docutils restructured text using S5 definitions rather
than Sphinx RST.
  
The generated S5 html and pdf are placed within the Sphinx
build directory at the appropriate place in the tree
corresponding to the RST sources


Creating screenshots from g4daeview.py
----------------------------------------

::

    slides-
    slides-screenshots-dir-cd
    g4daeview.sh --with-chroma --load 1


Include Fullscreen Image in S5 slides ?
----------------------------------------

* https://developer.mozilla.org/en-US/docs/Web/Guide/CSS/Scaling_background_images
* https://developer.mozilla.org/en-US/docs/Web/CSS/background
* https://developer.mozilla.org/en-US/docs/Web/CSS/background-repeat

Use raw html directive at the head of the RST source, identifying slides
to receive background images via div#id css selectors where the id are 
a somewhat mangled slide titles.

Note:

#. document relative and server relative links are usable from css
#. protocol relative, starting "//" also works but that would mean 
   hardcoding the sever hostname

::

    .. include:: <s5defs.txt>

    .. raw:: html

       <style type="text/css">

         /* 
              1282 × 960  pixels    143.99 pixels/inch  237.1 KB (242,821 bytes)     chroma_dayabay_adlid.png
              1278 × 962  pixels    143.99 pixels/inch  433.5 KB (443,928 bytes)     chroma_dayabay_pool_pmts.png      

              With "background-size: contain" and not specifying a size for the div leads 
              to scaling being dependant on the dimensions of the div, which depend on the amount 
              of content lines on the page, also this changes resize browser window. When
              little content the image is scaled up into top left corner.

              With "background-size: cover" and not specifying a size for the div leads 
              to scaling to fill horizontally, but vertical chop when the content ends.

              Omitting "background-size" and not specifying a size for the div get 
              no image scaling, it appears as-is but chopped according to size of the div. 

              Omitting "background-size" and specifying "div.slide { height: 100%; }"
              gets no image scaling and image presented as-is without any chopping.  This
              is a better approach as the per slide config is minimised to just 
              specifying the url. 

         */

          div.slide { 
             background-clip: border-box;
             background-repeat: no-repeat;
             height: 100%;
          }
          div.slide#full-screen{
             background-image: url(images/chroma/chroma_dayabay_adlid.png);
          }  
          div.slide#full-screen-2{
             background-image: url(images/chroma/chroma_dayabay_pool_pmts.png);
          }  
          div.slide#test-server-relative-link{
             background-image: url(/env/test/LANS_AD3_CoverGas_Humidity.png);
          }  
          div.slide#test-protocol-relative-link{
             background-image: url(//localhost/env/test/LANS_AD3_CoverGas_Humidity.png);
          }  


       </style>



       ...bulk of slides omitted...


       Full Screen
       ------------

       Full Screen 2
       ---------------

       Content appears on top of the image, that can be difficult to read.

      



Convert .html pages to .pdf 
------------------------------

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

PDF CREATION FUNCTIONS
------------------------

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
              uses convert to concatenate NN_crop.png into name.pdf , this 
              is using ImageMagick (a very heavy dependency). 

              Initially used Automator action, 
              /Library/WebServer/Documents/env/muon_simulation/presentation/pngs2pdf.workflow

              Subsequently created an Automator Service `Make PDF from PNGs`, 
              documented at :env:`osx/osx_automator_workflows`


*slides-rst2pdf-convert*
              DID NOT PURSUE THIS TECHNIQUE AS TOO MUCH STYLING REINVENTION


::

    delta:opticks_gpu_optical_photon_simulation blyth$ downsize.py *.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize 00_crop.png to create 00_crop_half.png 2682px_1498px -> 1341px_749px 
    INFO:env.doc.downsize:downsize 01_crop.png to create 01_crop_half.png 2682px_1498px -> 1341px_749px 
    INFO:env.doc.downsize:downsize 02_crop.png to create 02_crop_half.png 2682px_1498px -> 1341px_749px 
    ...
    INFO:env.doc.downsize:downsize 30_crop.png to create 30_crop_half.png 2682px_1498px -> 1341px_749px 
    INFO:env.doc.downsize:downsize 31_crop.png to create 31_crop_half.png 2682px_1498px -> 1341px_749px 
    INFO:env.doc.downsize:downsize 32_crop.png to create 32_crop_half.png 2682px_1498px -> 1341px_749px 


EOU
}

slides-env(){      elocal- ; bitbucketstatic- ; presentation- ;  }
slides-fold(){  echo $(slides-branch)/$(slides-name) ; }

#slides-dir(){   apache- ; echo $(apache-htdocs)/env/$(slides-fold) ; }
#slides-dir(){   echo $HOME/simoncblyth.bitbucket.io/env/$(slides-fold) ; }
slides-dir(){   echo /tmp/simoncblyth.bitbucket.io/env/$(slides-fold) ; }

slides-sdir(){  echo $(env-home)/$(slides-branch) ; } 
slides-pdir(){  echo $(env-home)/_build/dirhtml/$(slides-fold) ; }
slides-path(){  echo $(slides-dir)/$(slides-name).${1:-pdf} ; }
slides-open(){  open $(slides-path $*) ; } 
slides-info(){
   cat << EOI

   slides-fold : $(slides-fold)
   slides-dir  : $(slides-dir)
   slides-sdir : $(slides-sdir)             go here with slides-scd
   slides-pdir : $(slides-pdir)
   slides-path : $(slides-path)
   
   slides-name   : $(slides-name)          override via SLIDES_NAME $SLIDES_NAME
   slides-branch : $(slides-branch)        override via SLIDES_BRANCH $SLIDES_BRANCH 
   slides-host   : $(slides-host)          override via SLIDES_HOST $SLIDES_HOST
   slides-url    : $(slides-url)           override via SLIDES_URL $SLIDES_URL
   slides-ppath  : $(slides-ppath)

EOI
}

slides-ls(){  ls -l $(slides-dir); }
slides-cd(){  cd $(slides-dir)/$1; }
slides-scd(){  cd $(slides-sdir); }
slides-mkdir(){ mkdir -p $(slides-dir) ; }
slides-get(){

   # adjust safari window size
   slides-safari

   # capture safari window screenshots
   slides-capture $*

   # default crop.py style is safari_headtail, which removes the safari chrome
   slides-crop

   slides-rm-uncropped
   # on OSX invokes slided-convert-automator 
   # this just opens folder of .pngs and gives instructions on how to use the automator 
   # action to make .pdf from them
   slides-convert

}

slides-get-gtc(){       slides-get 0 42 ; }
slides-get-lecospa(){   slides-get 0 57 ; }
slides-get-jnu-cmake-ctest(){ slides-get 0 5 ; }
slides-get-llr(){     slides-get 0 32 ; }
slides-get-psroc(){   slides-get 0 26 ; }
slides-get-psroc0(){  slides-get 0 0 ; }
slides-get-jul2017(){ slides-get 0 34 ; }
slides-get-sdu(){ slides-get 0 64 ; }
#slides-get-sdu(){ slides-get 0 3 ; }
slides-get-sep2017wol(){ slides-get 0 47 ; }
slides-get-dybdb(){ slides-get 0 15 ; }
slides-get-sjtu(){ slides-get 0 22 ; }
slides-get-ihep(){ slides-get 0 42 ; }
slides-get-chep(){ slides-get 0 33 ; }


#slides-name(){      echo ${SLIDES_NAME:-gpu_optical_photon_simulation} ; }
#slides-name(){      echo ${SLIDES_NAME:-g4dae_geometry_exporter} ; }
slides-name(){       echo $(presentation-name) ; }
slides-branch(){    echo ${SLIDES_BRANCH:-presentation} ; }        # env relative path to where .txt sources reside

#slides-host(){      echo ${SLIDES_HOST:-dayabay.phys.ntu.edu.tw} ; }   
#slides-host(){      echo ${SLIDES_HOST:-simoncblyth.bitbucket.org} ; }   
slides-host(){      echo ${SLIDES_HOST:-localhost} ; }   

slides-url-prior(){ echo ${SLIDES_URL:-http://$(slides-host)/e/$(slides-fold)/$(slides-name).html} ; }
slides-ppath-prior(){ echo $(apache-htdocs $1)/e/$(slides-fold)/$(slides-name).${2:-pdf} ; }   

slides-url(){       echo ${SLIDES_URL:-http://$(slides-host)/env/$(slides-branch)/$(slides-name).html} ; }
slides-ppath(){     echo $(apache-htdocs $1)/env/$(slides-branch)/$(slides-name).${2:-pdf} ; }   

slides-url-page(){  echo "$(slides-url)?p=$1" ; }





slides-safari(){  osascript $(slides-safari-path) ; }
slides-safari-edit(){  vi $(slides-safari-path) ; }
slides-safari-path(){  echo $(env-home)/doc/safari.applescript ; }  
slides-chrome(){  osascript $(slides-chrome-path) ; }
slides-chrome-edit(){  vi $(slides-chrome-path) ; }
slides-chrome-path(){  echo $(env-home)/doc/chrome.applescript ; }  


slides-screenshots-dir(){ echo $(apache-htdocs)/env/geant4/geometry/collada/g4daeview ; }
slides-screenshots-dir-cd(){ cd $(slides-screenshots-dir) ; }


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
}

slides-rsync(){
   local msg="=== $FUNCNAME "
   slides-scd

   local outdir=$(slides-dir)
   local ans
   read -p "$msg rsync derived outputs to $outdir and clean them from working copy  : Enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 

   mkdir -p $outdir
   echo $msg rsync S5 html slides and sources to $outdir
   make rsync OUTDIR=$outdir

   echo $msg removing any derived .html .pdf files 
   make clean
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

#slides-fmt(){ echo pdf ; }  # PIL cannot crop PDF
slides-fmt(){ echo png ; }   
slides-quit(){ touch ~/QUIT ; }
slides-capture(){
   local msg="=== $FUNCNAME "

   slides-mkdir
   slides-cd

   local fmt=$(slides-fmt)
   local pages=$(slides-pages $1 $2)
   local page
   local url
   local zpage
   local name
   local cname
 

   for page in $pages ; do
      url=$(slides-url-page $page)
      zpage=$(printf "%0.2d" $page)
      name="${zpage}.${fmt}"
      cname="${zpage}_crop.${fmt}"

      [ -f "$HOME/QUIT" ] && echo $msg QUIT due to HOME/QUIT  && return 

      slides-safari

      if [ -f "$name" -o -f "$cname" ]; then 
          echo $msg file $name or $cname from url "$url" already downloaded : delete and rerun to refresh
      else

          echo $msg opening url "$url" 
          open "$url"
          cmd="screencapture -T0 -t$fmt -w -i -o $name"
          #
          #    -T<seconds>  Take the picture after a delay of <seconds>, default is 5 
          #    -w           only allow window selection mode
          #    -i           capture screen interactively, by selection or window
          #    -o           in window capture mode, do not capture the shadow of the window
          #    -t<format>   image format to create, default is png (other options include pdf, jpg, tiff and other formats)      
          #
          echo $msg about to do $cmd : tap browser window once loaded and highlighted blue
          sleep 1
          eval $cmd
      fi

   done
}
slides-crop(){
   local msg="=== $FUNCNAME "
   slides-cd
   echo $msg cropping png
   /opt/local/bin/python $ENV_HOME/bin/crop.py ??.png
}

slides-rm-uncropped(){
   slides-cd
   rm ??.png
}


slides-convert-automator(){
   local msg="=== $FUNCNAME "

   local dir=$(slides-dir)
   local cmd="open $dir"
   echo $msg : $cmd
   eval $cmd

   cat << EOD

Using *Make PDF from PNGs* Automator Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open folder containing PNGs to combine in Finder.app
#. Move uncropped to trash and arrange order of cropped by name and then select PNGs
#. ctrl-click the selection to get contextual menu, choose Make PDF from PNGs
#. After a second or so (depending on number of PNGs) a dialog will appear to enter the basename (without .pdf) of the output PDF.
#. After entering the name the new PDF will appear on the Desktop

For details see http://dayabay.phys.ntu.edu.tw/e/osx/osx_automator_workflows/

After checking the resulting PDF move the prior one aside and
copy the new in its place, and sync across to remote server::

    delta:presentation blyth$ pwd
    /Library/WebServer/Documents/env/muon_simulation/presentation
    delta:presentation blyth$ mv gpu_optical_photon_simulation.pdf gpu_optical_photon_simulation_apr28.pdf
    delta:presentation blyth$ mv ~/Desktop/gpu_optical_photon_simulation.pdf .
    delta:presentation blyth$ env-htdocs-rsync



EOD

}
slides-convert(){
   local msg="=== $FUNCNAME "
   local pdf=$(slides-name).pdf   
   slides-cd
   echo $msg converting PNG into $pdf 

   [ "$(which convert)" == "" ] && slides-convert-automator && return 

   convert ??_crop.png $pdf
}
slides-scp(){
   local msg="=== $FUNCNAME "
   local tag=${1:-C2R}
   local scp="scp $(slides-path pdf) $tag:$(NODE_TAG=$tag slides-path pdf)"
   echo $msg $scp 
   eval $scp
   echo $msg : NB may need to do a slides-publish on the destination to make the PDF accessible
}

slides-scp-htdocs(){
   # apache on C2R resisting serving the pdf
   local msg="=== $FUNCNAME "
   local tag=${1:-C2R}
   apache-
   local scp="scp $(slides-path pdf) $tag:$(slides-ppath $tag pdf)"
   echo $msg $scp 
   eval $scp
}


slides-rst2pdf(){
   /opt/local/Library/Frameworks/Python.framework/Versions/2.6/bin/rst2pdf $* 
}
slides-rst2pdf-convert(){
  local name=$(slides-name)
  #slides-rst2pdf $name.txt -o $name.pdf
  slides-rst2pdf $name.txt -b1 -s slides.style -o $name.pdf
}



slides-apache-prepare(){
   local msg="=== $FUNCNAME "
   apache- 
   local cmd="sudo mkdir -p $(apache-htdocs)/env/$(slides-branch)"
   echo $msg $cmd
   eval $cmd
   local GROUP=$(id -gn)
   cmd="sudo chown -R $USER:$GROUP $(apache-htdocs)/env"
   echo $msg $cmd
   eval $cmd
}

slides-apache-publish(){
   local iwd=$PWD
   local target=$(apache-htdocs)/env/$(slides-branch)

   cd $(env-home)/$(slides-branch)

   # THIS IS NOW DONE IN THE MAKEFILE
   #local cmd="cp $(slides-name).html $target/"
   #echo $cmd
   #eval $cmd

   local cmd
   local dirs="ui images"
   for dir in $dirs ; do 
       cmd="cp -r $dir $target/"
       echo $cmd
       eval $cmd
   done

   cd $iwd
   cmd="open http://localhost/env/$(slides-branch)/$(slides-name).html"
   echo $cmd
   eval $cmd

}



