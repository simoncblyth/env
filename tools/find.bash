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

