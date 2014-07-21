# === func-gen- : hg/hgapi fgp hg/hgapi.bash fgn hgapi fgh hg
hgapi-src(){      echo hg/hgapi.bash ; }
hgapi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hgapi-src)} ; }
hgapi-vi(){       vi $(hgapi-source) ; }
hgapi-env(){      elocal- ; }
hgapi-usage(){ cat << EOU

HGAPI
=======

https://bitbucket.org/haard/hgapi
https://bitbucket.org/haard/autohook

hgapi is a pure-Python API to Mercurial, that uses the command-line interface
instead of the internal Mercurial API. The rationale for this is twofold: the
internal API is unstable, and it is GPL



EOU
}
hgapi-dir(){ echo $(local-base)/env/hg/hg-hgapi ; }
hgapi-cd(){  cd $(hgapi-dir); }
hgapi-mate(){ mate $(hgapi-dir) ; }
hgapi-get(){
   local dir=$(dirname $(hgapi-dir)) &&  mkdir -p $dir && cd $dir

    hg clone https://bitbucket.org/haard/hgapi

}
