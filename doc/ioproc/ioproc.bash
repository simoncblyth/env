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


ioproc-conf(){ echo chep2016 ; }
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
ioproc-etex(){ echo $(ioproc-edir)/$(ioproc-conf).tex ; }
ioproc-edit(){ vi $(ioproc-etex) ; }
ioproc-make(){

   ioproc-cd
   local conf=$(ioproc-conf)
   local etex=$(ioproc-etex) 

   local tex=$conf.tex
   local dvi=$conf.dvi
   local pdf=$conf.pdf

   [ $etex -nt $tex ] && echo $msg UPDATING FROM ETEX $etex && cp $etex $tex || return 

   latex $tex > /dev/null
   dvipdf $dvi > /dev/null  

   ls -l $pdf
}

ioproc--()
{
    ioproc-make
    ioproc-open 
}



