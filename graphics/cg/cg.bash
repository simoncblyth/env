# === func-gen- : graphics/cg/cg fgp graphics/cg/cg.bash fgn cg fgh graphics/cg
cg-src(){      echo graphics/cg/cg.bash ; }
cg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cg-src)} ; }
cg-vi(){       vi $(cg-source) ; }
cg-env(){      elocal- ; }
cg-usage(){ cat << EOU

CG TOOLKIT : Cg a C-like graphics language
============================================

* https://developer.nvidia.com/cg-toolkit-download

OSX
-----

PPC still supported, installer is terse in the extreme

* http://developer.download.nvidia.com/cg/Cg_3.1/Cg-3.1_April2012.dmg
* http://developer.download.nvidia.com/cg/Cg_3.1/Cg-3.1_April2012_ReferenceManual.pdf

::

    simon:~ blyth$ du -h /Library/Frameworks/Cg.framework
     39M    /Library/Frameworks/Cg.framework

    simon:~ blyth$ which cgc
    /usr/bin/cgc
    simon:~ blyth$ cgc -h
    Usage: cgc [options] file 

    Options: 

            ---------- Basic Command Line Options ----------   
               [-entry id | -noentry] [-o ofile] [-l lfile] 
               [-profile id] [-po|-profileopts opt1,opt2,...]
     


EOU
}
cg-dir(){ echo $(local-base)/env/graphics/cg/graphics/cg-cg ; }
cg-cd(){  cd $(cg-dir); }
cg-mate(){ mate $(cg-dir) ; }
cg-get(){
   local dir=$(dirname $(cg-dir)) &&  mkdir -p $dir && cd $dir

}
