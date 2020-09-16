# === func-gen- : python/py3/py3 fgp python/py3/py3.bash fgn py3 fgh python/py3 src base/func.bash
py3-source(){   echo ${BASH_SOURCE} ; }
py3-edir(){ echo $(dirname $(py3-source)) ; }
py3-ecd(){  cd $(py3-edir); }
py3-dir(){  echo $LOCAL_BASE/env/python/py3/py3 ; }
py3-cd(){   cd $(py3-dir); }
py3-vi(){   vi $(py3-source) ; }
py3-env(){  elocal- ; }
py3-usage(){ cat << EOU

Notes on py3 changes
=======================


commands module has gone
-------------------------

::
 
    try:
        from commands import getstatusoutput 
    except ImportError:
        from subprocess import getstatusoutput 
    pass 
           






EOU
}
py3-get(){
   local dir=$(dirname $(py3-dir)) &&  mkdir -p $dir && cd $dir

}
