# === func-gen- : apache/modroot fgp apache/modroot.bash fgn modroot
modroot-src(){      echo apache/modroot.bash ; }
modroot-source(){   echo ${BASH_SOURCE:-$(env-home)/$(modroot-src)} ; }
modroot-vi(){       vi $(modroot-source) ; }
modroot-env(){      apache- ; }
modroot-usage(){
  cat << EOU
     modroot-src : $(modroot-src)
    $(env-wikiurl)/ModRoot

EOU
}

modroot-url(){ echo ftp://root.cern.ch/root/mod_root2.c ; }
modroot-dir(){ echo $(local-base)/env/apache/modroot ; }
modroot-get(){
   local dir=$(modroot-dir)
   mkdir -p $dir
   cd $dir
   [ ! -f $(basename $(modroot-url)) ] && curl -O $(modroot-url) 
}
modroot-cd(){ cd $(modroot-dir) ; }
modroot-build(){
   local msg="=== $FUNCNAME :"
   [ "$(which apxs)" == "" ] && echo $msg ERROR no apxs && return 1 
   modroot-cd
   echo $msg using $(which apxs) in $PWD
   local cmd="apxs -c mod_root2.c"
   echo $msg $cmd
   eval $cmd
}

modroot-install(){
   local msg="=== $FUNCNAME :"
   [ "$(which apxs)" == "" ] && echo $msg ERROR no apxs && return 1 
   modroot-cd
   echo $msg using $(which apxs) in $PWD
   local cmd="sudo apxs -i -c mod_root2.c"
   echo $msg $cmd
   eval $cmd
}

modroot-conf-(){ cat << EOC
#
# $(modroot-source) $FUNCNAME $(date)
# see wiki:ModRoot and ftp://root.cern.ch/root/mod_root2.c
#
LoadModule root_module modules/mod_root2.so
AddHandler mod-root2 .root
AddHandler mod-root2 .zip

# the last line is only needed if you want to support ZIP files containing
EOC
}

modroot-conf-path(){  echo $(dirname $(apache-conf))/modroot.conf ; }

modroot-conf(){
  local msg="=== $FUNCNAME :"
  local cnf=$($FUNCNAME-path)
  local tmp=/tmp/env/apache/$FUNCNAME/$(basename $cnf) ; 
  mkdir -p $(dirname $tmp)
  $FUNCNAME- > $tmp

  local cmd="sudo cp $tmp $cnf " 
  echo $msg $cmd 
  eval $cmd  

}

modroot-append(){
  echo Include conf/$(basename $(modroot-conf-path)) >> $(apache-conf)

}




modroot-upload(){
   local path=${1:-run00027.root}
   local    t=${2:-C2}
   scp $path $t:$(apache-htdocs $t)/aberdeen/
}


modroot-test(){ $FUNCNAME- | python ; }
modroot-test-(){ cat << EOT
import os
os.environ["LD_LIBRARY_PATH"] += ":" + os.path.join( os.environ["HOME"] , "aberdeen/DataModel/lib" ) 
from ROOT import gSystem, TFile
gSystem.Load("libAbtDataModel")
f = TFile.Open("http://dayabay.phys.ntu.edu.tw/aberdeen/run00027.root");
t = f.Get("T")  
print t.GetEntries()
EOT
}



