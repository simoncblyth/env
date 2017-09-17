# === func-gen- : doc/ioproc/ioproc fgp doc/ioproc/ioproc.bash fgn ioproc fgh doc/ioproc
ioproc-src(){      echo doc/ioproc/ioproc.bash ; }
ioproc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ioproc-src)} ; }
ioproc-vi(){       vi $(ioproc-source) ; }
ioproc-env(){      elocal- ; }
ioproc-usage(){ cat << EOU

IOP Conference Proceedings Series
=====================================

* http://conferenceseries.iop.org/content/authors
* http://chep2016.org/node/28


Submission

* https://indico.cern.ch/event/505613/contributions/

228. Opticks : GPU Optical Photon Simulation for Particle Physics with NVIDIA OptiX
 simon blyth
 11/10/2016, 12:15

When submitting your paper, you need to know its Indico ID. 


* https://chep2016.conferenceseries.iop.org/


CHEP2016 submission
----------------------

Dear Simon
Thank you for submitting the Full Paper titled "Opticks : GPU Optical Photon Simulation for Particle Physics using NVIDIA OptiX". 
The paper number of your submission is CHEP201700016. 
Your submission will now be passed to the editor.

Dear Simon,

The paper entitled 'Opticks : GPU Optical Photon Simulation for Particle Physics using NVIDIA OptiX' will now be passed to the editors.
You will be able to check on the progress of your paper by logging on to submission system:

https://chep2016.conferenceseries.iop.org/pdf-submitted/



JUNO
-----

Dear Publication board members and colleagues,

My CHEP 2016 proceedings paper is uploaded to:
      http://juno.ihep.ac.cn/cgi-bin/Dev_DocDB/ShowDocument?docid=2100

This technical talk/proceedings was not “On behalf of the JUNO Collaboration” .

The deadline for CHEP 2016 proceedings is Monday February 6th, PST(-16),
slightly under 3 weeks from today. 
I welcome any comments, especially ones prior to Chinese New Year.

Simon

DYB
----

Dear Publication board members and colleagues,

My CHEP 2016 proceedings paper is uploaded to:
     http://dayabay.ihep.ac.cn/cgi-bin/DocDB/ShowDocument?docid=11266

This technical talk/proceedings was not “On behalf of the Daya Bay Collaboration”

The deadline for CHEP 2016 proceedings is Monday February 6th, PST(-16),
slightly under 3 weeks from today. 
I welcome any comments, especially ones prior to Chinese New Year.

Simon


Dyb
----

* http://dayawane.ihep.ac.cn/twiki/pub/Internal/WebHome/DayaBay_bylaws16.pdf
* http://dayabay.ihep.ac.cn/DocDB/0112/011232/003/PubCommReport_20161210_v3.pdf


Previous Years Proceedings
----------------------------

21st International Conference on Computing in High Energy and Nuclear Physics (CHEP2015), Japan

* http://iopscience.iop.org/volume/1742-6596/664
* http://iopscience.iop.org/issue/1742-6596/664/7

chep2016
----------

* Submission Deadline Monday February 6, 2017
* All 15-minute Oral Presentations and Poster Presentations have a limit of 8 pages.

         Taipei (+16)                San Fransisco                 
   [Mon] 2305, 06 Feb 2017       [Mon] 0705, 06 Feb 2017 
   [Tue] 0110, 07 Feb 2017       [Mon] 0910, 06 Feb 2017
   [Tue] 0400, 07 Feb 2017       [Mon] 1210, 06 Feb 2017 
   [Tue] 1000, 07 Feb 2017       [Mon] 1800, 06 Feb 2017 
   [Tue] 1500, 07 Feb 2017       [Mon] 2300, 06 Feb 2017
   [Tue] 1601, 07 Feb 2017       [Tue] 0001, 07 Feb 2017       

* Assuming submissions are allowed on the San Francisco day of the deadline, 
  have until 16:00 on Taipei's Tuesday 

See: presentation-writeup for planning and text creation


Initialize .tex sources in env
---------------------------------

Copy JPCSLaTeXGuidelines.tex into the ioproc-dir 
named after the conference, renaming to eg chep2016.tex

::

    simon:chep2016 blyth$ cp JPCSLaTeXGuidelines.tex ~/e/doc/ioproc/chep2016/chep2016.tex



Screenshot Figure Prep 
---------------------------

#. shift-cmd-4 take screen shot saving onto desktop       

#. copy renamed png into bitbucket static folders

::
 
    cd ~/opticks/ok
    osx_
    osx_ss_cp dyb_raytrace_composite_cerenkov   

    # relative path of invoking directory within opticks or env 
    # determines the destination folder within bitbucket statics ~/simoncblyth.bitbucket.org 

#. downsize the large screen shot twice 

::

    In [2]: 2862.*1688./1e6
    Out[2]: 4.831056
  
    cd ~/simoncblyth.bitbucket.org/env/ok/
    downsize.py dyb_raytrace_composite_cerenkov.png          # 2862px_1688px -> 1431px_844px
    downsize.py dyb_raytrace_composite_cerenkov_half.png     # 1431px_844px -> 715px_422px 

pdflatex is complaining::

   libpng warning: iCCP: known incorrect sRGB profile

Tried with the original screen shot, and get same warning and a much too big image.

* http://stackoverflow.com/questions/11041044/convert-jpg-from-adobergb-to-srgb-using-pil


Get warning to go away by assigning profile with Preview::

    cp dyb_raytrace_composite_cerenkov_half_half.png dyb_raytrace_composite_cerenkov_half_half_assign_profile.png

    open dyb_raytrace_composite_cerenkov_half_half_assign_profile.png

    ## in Preview>Tools>Assign Profile..  picked "Generic RGB" 



Initially had some trouble controlling size of includgraphics in latex,
using resolution seems the easiest way.

* http://tex.stackexchange.com/questions/12939/png-importing-it-with-latex-pdflatex-xelatex
* http://tex.stackexchange.com/questions/21627/image-from-includegraphics-showing-in-wrong-image-size

% Opening in Preview shows 640x360px 72px/inch 

%\includegraphics[natwidth=640bp,natheight=360bp,resolution=500]{jpmt-inside-wide_crop_half_half.png}
%\includegraphics[angle=45]{jpmt-inside-wide.png}




FUNCTIONS
----------

*ioproc-edit*
     edit env .tex sources 

*ioproc--*
     run latex and dvipdf producing pdf from tex sources, and open the pdf 



EOU
}


ioproc-conf(){ echo ${IOPROC_CONF:-chep2016} ; }
ioproc-pdfname(){ echo ${IPPROC_PDFNAME:-opticks-blyth-chep2016} ; }

ioproc-dir(){ echo $(local-base)/env/doc/ioproc/$(ioproc-conf) ; }
ioproc-edir(){ echo $(env-home)/doc/ioproc/$(ioproc-conf) ; }
ioproc-cd(){  cd $(ioproc-dir); }
ioproc-ecd(){ cd $(ioproc-edir); }

ioproc-url(){ echo http://cms.iopscience.iop.org/alfresco/d/d/workspace/SpacesStore/a83f1ab6-cd8f-11e0-be51-5d01ae4695ed/LaTeXTemplates.zip ; }
ioproc-get(){
   local dir=$(dirname $(ioproc-dir)) &&  mkdir -p $dir && cd $dir


   local url=$(ioproc-url) 
   local nam=$(basename $url)
   local conf=$(ioproc-conf)

   [ ! -f $nam ] && curl -L -O $url 
   [ ! -d $conf ] && unzip -d $conf $nam

   local edir=$(ioproc-edir)
   [ ! -d "$edir" ] && mkdir -p $edir
}


ioproc-pdf(){  echo $(ioproc-dir)/$(ioproc-pdfname).pdf ; }
ioproc-open(){ open $(ioproc-pdf) ; }
ioproc-guide(){ open $(ioproc-dir)/JPCSLaTeXGuidelines.pdf ; }

ioproc-etex(){ echo $(ioproc-edir)/$(ioproc-conf).tex ; }
ioproc-edit(){ vi $(ioproc-etex) ; }
ioproc-e(){ vi $(ioproc-etex) ; }

ioproc-make-verbose(){ export VERBOSE=1 ; ioproc-make ; }

ioproc-make0-(){
  local tex=$1
  local dvi=${tex/.tex}.dvi
  if [ -n "$VERBOSE" ]; then
      latex $tex 
      latex $tex 
      latex $tex 
      dvipdf $dvi
  else
      latex $tex > /dev/null
      latex $tex > /dev/null
      latex $tex > /dev/null
      dvipdf $dvi > /dev/null  
  fi 
}

ioproc-make1-(){
 # http://tex.stackexchange.com/questions/17734/cannot-determine-size-of-graphic
  local tex=$1
  if [ -n "$VERBOSE" ]; then
      pdflatex $tex 
      pdflatex $tex 
      pdflatex $tex 
  else
      pdflatex $tex > /dev/null
      pdflatex $tex > /dev/null
      pdflatex $tex > /dev/null
  fi 
}

ioproc-make(){

   ioproc-cd
   local conf=$(ioproc-conf)
   local etex=$(ioproc-etex) 

   local pdfname=$(ioproc-pdfname)

   local tex=$pdfname.tex
   local pdf=${tex/.tex}.pdf

   [ $etex -nt $tex ] && echo $msg UPDATING FROM ETEX $etex && cp $etex $tex || return 

   echo tex $tex

   #ioproc-make0- $tex
   ioproc-make1- $tex 

   ls -l $pdf
   du -h $pdf
}

ioproc--()
{
    ioproc-make
    ioproc-open 
}

ioproc-info(){ cat << EOI

   ioproc-conf     : $(ioproc-conf)
   ioproc-etex     : $(ioproc-etex)
   ioproc-pdfname  : $(ioproc-pdfname)
   ioproc-pdf      : $(ioproc-pdf)

EOI

    ls -l $(ioproc-pdf)

}


ioproc-figs-(){ cat << EOF
$HOME/simoncblyth.bitbucket.org/env/ok/dyb_raytrace_composite_cerenkov.png
$HOME/simoncblyth.bitbucket.org/env/ok/dyb_raytrace_composite_cerenkov_half.png
$HOME/simoncblyth.bitbucket.org/env/ok/dyb_raytrace_composite_cerenkov_half_half.png
$HOME/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop.png
$HOME/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop_half.png
$HOME/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop_half_half.png
EOF
}

ioproc-figs-copy(){

  local dstd=$(ioproc-dir)

  local msg="=== $FUNCNAME :"
  local src
  ioproc-figs- | while read src ; do
     [ ! -f "$src" ] && echo $msg src $src does not exist && return 
     local nam=$(basename $src)
     local dst=$dstd/$nam

     if [ -f "$dst" ]; then  
         echo $msg dst $dst already copied 
     else
         cp $src $dst 
     fi
  done

}

