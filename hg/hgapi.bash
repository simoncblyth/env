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

Seealso *adm-vi*


API Example
-------------

::

    import hgapi
    env = hgapi.Repo("/tmp/mercurial/env")
    for r in env[0:100]:print "%10s %s " % (r.rev,r.desc)



EOU
}
hgapi-dir(){ echo $(local-base)/env/hg/hg-hgapi ; }
hgapi-cd(){  cd $(hgapi-dir); }
hgapi-mate(){ mate $(hgapi-dir) ; }
hgapi-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(hgapi-dir)) &&  mkdir -p $dir && cd $dir

   [ -z "$VIRTUAL_ENV" ] && echo $msg requires VIRTUAL_ENV : do adm- && return
   [ "$(basename $VIRTUAL_ENV)" != "adm_env" ] && echo $msg NOT IN ADM ENV DO adm- FIRST && return 

   pip -v install hgapi
}

hgapi-install(){
   adm-
   which python
   hgapi-get
}


