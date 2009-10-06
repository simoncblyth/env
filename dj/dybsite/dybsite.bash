# === func-gen- : dj/dybsite/dybsite fgp dj/dybsite/dybsite.bash fgn dybsite fgh dj/dybsite
dybsite-src(){      echo dj/dybsite/dybsite.bash ; }
dybsite-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dybsite-src)} ; }
dybsite-vi(){       vi $(dybsite-source) ; }
dybsite-env(){      elocal- ; }
dybsite-usage(){
  cat << EOU

     dybsite- specialization of dj- 
    =================================

     dybsite-dir : $(dybsite-dir)

     $(env-wikiurl)/OfflineDB


EOU
}
dybsite-dir(){ echo $(local-base)/env/dj/dybsite ; }
dybsite-cd(){  cd $(dybsite-dir); }
dybsite-mate(){ mate $(dybsite-dir) ; }
dybsite-get(){
   local dir=$(dirname $(dybsite-dir)) &&  mkdir -p $dir && cd $dir

}
## models introspection from db ##  

dybsite-models(){
   local msg="=== $FUNCNAME :"
   echo $msg creating $($FUNCNAME-path)
   local path=$(dj-models-path)
   mkdir -p $(dirname $path) 
   touch $(dirname $path)/__init__.py
   $FUNCNAME-inspectdb > $path
}

dybsite-models-path(){  echo $(dj- ; dj-appdir)/generated/models.py ; }
dybsite-models-inspectdb(){ dj- ; dj-manage- inspectdb | python $(dj-dir)/fix.py ; }

dybsite-check-settings(){
    type $FUNCNAME
    apache-
    ## python -c "import dybsite.settings " should fail with permission denied 
    sudo -u $(apache-user) python -c "import dybsite.settings "
    sudo -u $(apache-user)  python -c "import dybsite.settings as s ; print '\n'.join(['%s : %s ' % ( v, getattr(s, v) ) for v in dir(s) if v.startswith('DATABASE_')]) "
}

dybsite-build(){

   dj-
   dj-build

   ## load from mysqldump 
   offdb-
   offdb-build

   ## introspect the db schema to generate and fix models.py
   dybsite-models

   dj-ip-

}


