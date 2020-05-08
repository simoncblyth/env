# === func-gen- : tools/find fgp tools/find.bash fgn find fgh tools
find-src(){      echo tools/find.bash ; }
find-source(){   echo ${BASH_SOURCE:-$(env-home)/$(find-src)} ; }
find-vi(){       vi $(find-source) ; }
find-env(){      elocal- ; }
find-usage(){ cat << EOU

Find things
=============

find-tabs py
     lists all .py files containing tables beneath invoking directory 


EOU
}
find-dir(){ echo $(local-base)/env/tools/tools-find ; }
find-cd(){  cd $(find-dir); }
find-mate(){ mate $(find-dir) ; }


find-tabs(){
   local typ=${1:-py}
   find . -name "*.$typ" -exec grep -l $'\t' {} \;
}

find-1m(){
  find . -name 'ht.npy' -size +1M

}

find-without-dir(){
  find $JUNOTOP/ExternalLibs  -path $JUNOTOP/ExternalLibs/Build -prune -o -name BoostConfig.cmake
}


find-1m-tot(){
   #find . -name 'ht.npy' -size +1M -print0 | xargs -0 du -hc | tail -n1
   # runs into too many arguments for du, so it gets run multiple times and so the last is far too small a total 
   find . -name 'ht.npy' -size +1M -print0 | du --files0-from=- -hc | tail -n1
}



