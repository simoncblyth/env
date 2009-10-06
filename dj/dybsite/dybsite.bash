# === func-gen- : dj/dybsite/dybsite fgp dj/dybsite/dybsite.bash fgn dybsite fgh dj/dybsite
dybsite-src(){      echo dj/dybsite/dybsite.bash ; }
dybsite-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dybsite-src)} ; }
dybsite-vi(){       vi $(dybsite-source) ; }
dybsite-env(){      elocal- ; }
dybsite-usage(){
  cat << EOU
     dybsite-src : $(dybsite-src)
     dybsite-dir : $(dybsite-dir)


EOU
}
dybsite-dir(){ echo $(local-base)/env/dj/dybsite/dj/dybsite-dybsite ; }
dybsite-cd(){  cd $(dybsite-dir); }
dybsite-mate(){ mate $(dybsite-dir) ; }
dybsite-get(){
   local dir=$(dirname $(dybsite-dir)) &&  mkdir -p $dir && cd $dir

}



dybsite-check-settings(){
type $FUNCNAME
apache-
## python -c "import dybsite.settings " should fail with permission denied 

sudo -u $(apache-user) python -c "import dybsite.settings "
sudo -u $(apache-user)  python -c "import dybsite.settings as s ; print '\n'.join(['%s : %s ' % ( v, getattr(s, v) ) for v in dir(s) if v.startswith('DATABASE_')]) "
}





