# === func-gen- : doc/ioproc/ioproc fgp doc/ioproc/ioproc.bash fgn ioproc fgh doc/ioproc
ioproc-src(){      echo doc/ioproc/ioproc.bash ; }
ioproc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ioproc-src)} ; }
ioproc-vi(){       vi $(ioproc-source) ; }
ioproc-env(){      elocal- ; }
ioproc-usage(){ cat << EOU

IOP Conference Proceedings Series
=====================================

* http://conferenceseries.iop.org/content/authors
* http://chep2016.org/node/24



Previous Years Proceedings
----------------------------

21st International Conference on Computing in High Energy and Nuclear Physics (CHEP2015), Japan

* http://iopscience.iop.org/volume/1742-6596/664
* http://iopscience.iop.org/issue/1742-6596/664/7


chep2016
----------

* Submission Deadline Monday February 6, 2017
* All 15-minute Oral Presentations and Poster Presentations have a limit of 8 pages.

See: presentation-writeup for planning and text creation


Initialize .tex sources in env
---------------------------------

Copy JPCSLaTeXGuidelines.tex into the ioproc-dir 
named after the conference, renaming to eg chep2016.tex

::

    simon:chep2016 blyth$ cp JPCSLaTeXGuidelines.tex ~/e/doc/ioproc/chep2016/chep2016.tex


FUNCTIONS
----------

*ioproc-edit*
     edit env .tex sources 

*ioproc--*
     run latex and dvipdf producing pdf from tex sources, and open the pdf 



EOU
}


ioproc-conf(){ echo ${IOPROC_CONF:-chep2016} ; }
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


ioproc-pdf(){  echo $(ioproc-dir)/$(ioproc-conf).pdf ; }
ioproc-open(){ open $(ioproc-pdf) ; }
ioproc-guide(){ open $(ioproc-dir)/JPCSLaTeXGuidelines.pdf ; }

ioproc-etex(){ echo $(ioproc-edir)/$(ioproc-conf).tex ; }
ioproc-edit(){ vi $(ioproc-etex) ; }

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

   local tex=$conf.tex
   local pdf=${tex/.tex}.pdf

   [ $etex -nt $tex ] && echo $msg UPDATING FROM ETEX $etex && cp $etex $tex || return 

   echo tex $tex

   #ioproc-make0- $tex
   ioproc-make1- $tex

   ls -l $pdf
}

ioproc--()
{
    ioproc-make
    ioproc-open 
}

ioproc-info(){ cat << EOI

   ioproc-conf : $(ioproc-conf)
   ioproc-etex : $(ioproc-etex)
   ioproc-pdf  : $(ioproc-pdf)


simon:chep2016 blyth$ l $HOME/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide*
-rw-r--r--@ 1 blyth  staff   622872 Dec 19 14:00 /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop_half_half.png
-rw-r--r--  1 blyth  staff  2285892 May  8  2016 /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop_half.png
-rw-r--r--@ 1 blyth  staff  6601014 Mar 26  2016 /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_crop.png
-rw-r--r--  1 blyth  staff  2926418 Jan 13  2016 /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide_half.png
-rw-r--r--@ 1 blyth  staff  9162714 Jan 13  2016 /Users/blyth/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide.png
simon:chep2016 blyth$ 


    Make a dyb plot with ray trace sliced composited with raster ?

        op -c --analyticmesh 1   # analytic PMTs

        /Users/blyth/opticks/oglrap/gl/tex/frag.glsl

::

 15 void main ()
 16 {
 17    frag_colour = texture(ColorTex, texcoord);
 18    float depth = frag_colour.w ;  // alpha is hijacked for depth in pinhole_camera.cu material1_radiance.cu
 19    frag_colour.w = 1.0 ;
 20 
 21    gl_FragDepth = depth  ;
 22 
 23    if(NrmParam.z == 1)
 24    {
 25         if(depth < ScanParam.x || depth > ScanParam.y ) discard ;
 26    }



EOI
}


ioproc-figs-(){ cat << EOF
$HOME/simoncblyth.bitbucket.org/env/graphics/ggeoview/jpmt-inside-wide.png
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


