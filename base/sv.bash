# === func-gen- : base/sv fgp base/sv.bash fgn sv fgh base
sv-src(){      echo base/sv.bash ; }
sv-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sv-src)} ; }
sv-vi(){       vi $(sv-source) ; }
sv-env(){      elocal- ; }
sv-usage(){
  cat << EOU
     sv-src : $(sv-src)
     sv-dir : $(sv-dir)



       http://supervisord.org/manual/current/


    Bootstrap the conf with 
        echo_supervisord_conf > $(sv-confpath)



EOU
}
sv-dir(){      echo $(local-base)/env/sv ; }
sv-confpath(){ echo $(sv-dir)/supervisord.conf ; }
sv-cd(){  cd $(sv-dir); }
sv-mate(){ mate $(sv-dir) ; }
sv-get(){
   #local dir=$(dirname $(sv-dir)) &&  mkdir -p $dir && cd $dir
   which python
   which easy_install
   easy_install supervisor
}

sv-bootstrap(){
   local conf=$(sv-confpath)
   mkdir -p $(dirname $conf) 
   echo_supervisord_conf > $conf
}

sv-edit(){ vim $(sv-confpath) ; }
sv-start(){ supervisord   -c $(sv-confpath) -n $* ; }
sv-ctl(){   supervisorctl -c $(sv-confpath) $* ; }






