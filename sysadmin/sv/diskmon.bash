# === func-gen- : sysadmin/sv/diskmon fgp sysadmin/sv/diskmon.bash fgn diskmon fgh sysadmin/sv
diskmon-src(){      echo sysadmin/sv/diskmon.bash ; }
diskmon-source(){   echo ${BASH_SOURCE:-$(env-home)/$(diskmon-src)} ; }
diskmon-vi(){       vi $(diskmon-source) ; }
diskmon-env(){      elocal- ; }
diskmon-usage(){
  cat << EOU
     diskmon-src : $(diskmon-src)
     diskmon-dir : $(diskmon-dir)


EOU
}
diskmon-dir(){ echo $(local-base)/env/sysadmin/sv/sysadmin/sv-diskmon ; }
diskmon-cd(){  cd $(diskmon-dir); }
diskmon-mate(){ mate $(diskmon-dir) ; }
diskmon-get(){
   local dir=$(dirname $(diskmon-dir)) &&  mkdir -p $dir && cd $dir
}

diskmon-dir(){ echo $(dirname $(diskmon-source)) ; }
diskmon-path(){ echo $(diskmon-dir)/diskmon.py ; }

diskmon-sv-(){ cat << EOL
[eventlistener:diskmon]
command=$(diskmon-path) -d /data -p 90 -m $(local-email)
events=TICK_60
EOL
}
