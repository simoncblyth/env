# === func-gen- : recon/minuit2 fgp recon/minuit2.bash fgn minuit2 fgh recon src base/func.bash
minuit2-source(){   echo ${BASH_SOURCE} ; }
minuit2-edir(){ echo $(dirname $(minuit2-source)) ; }
minuit2-ecd(){  cd $(minuit2-edir); }
minuit2-dir(){  echo $LOCAL_BASE/env/recon/Minuit2 ; }
minuit2-cd(){   cd $(minuit2-dir); }
minuit2-vi(){   vi $(minuit2-source) ; }
minuit2-env(){  elocal- ; }
minuit2-usage(){ cat << EOU

Minuit2 
=========

* https://root.cern.ch/doc/v608/Minuit2Page.html

A standalone version of Minuit2 (independent of ROOT) can be downloaded from

* https://root.cern.ch/doc/Minuit2.tar.gz  

  * BUT THATS A BROKEN LINK 

It does not contain the ROOT interface and it is therefore totally
independent of external packages and can be simply build using the configure
script and then make. Example tests are provided in the directory test/MnSim
and test/MnTutorial and they can be built with the make check command. The
Minuit2 User Guide provides all the information needed for using directly
(without add-on packages like ROOT) Minuit2.

* https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html

* https://github.com/jpivarski/pyminuit2

* http://seal.web.cern.ch/seal/work-packages/mathlibs/minuit/release/download.html


INSTEAD GET THIS ONE

5.34.14	
2014/01/24

* http://www.cern.ch/mathlibs/sw/5_34_14/Minuit2/Minuit2-5.34.14.tar.gz



EOU
}
minuit2-get(){
   local dir=$(dirname $(minuit2-dir)) &&  mkdir -p $dir && cd $dir

   #local url=https://root.cern.ch/doc/Minuit2.tar.gz
   local url=http://www.cern.ch/mathlibs/sw/5_34_14/Minuit2/Minuit2-5.34.14.tar.gz
   local dst=$(basename $url) 
   local nam=$(echo ${dst/.tar.gz})

   [ ! -f "$dst" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $dst 
   ln -svf $nam Minuit2 
}

minuit2-build()
{
   minuit2-cd
   ./configure --prefix=$(minuit2-dir).install
   make
   make install
}

minuit2-build-notes(){ cat << EON

----------------------------------------------------------------------
Libraries have been installed in:
   /usr/local/env/recon/Minuit2.install/lib

EON
}

minuit2--()
{
   minuit2-get
   minuit2-build
}

minuit2-check()
{
   # https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html#what-m-is-intended-to-do
   minuit2-cd
   make check  
   ./test/MnTutorial/test_Minuit2_Quad4FMain
}
