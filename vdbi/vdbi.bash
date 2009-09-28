# === func-gen- : vdbi/vdbi fgp vdbi/vdbi.bash fgn vdbi fgh vdbi
vdbi-src(){      echo vdbi/vdbi.bash ; }
vdbi-source(){   echo ${BASH_SOURCE:-$(env-home)/$(vdbi-src)} ; }
vdbi-vi(){       vi $(vdbi-source) ; }
vdbi-env(){      elocal- ; }
vdbi-usage(){
  cat << EOU
     vdbi-src : $(vdbi-src)
     vdbi-dir : $(vdbi-dir)

     vdbi-build
         installs dependencies, Rum, RumAlchemy,tw.rum

     vdbi-extras
        get and install ToscaWidgets + tw.jquery from mercurial repo
        they dont have recent enough pypi releases to fit into the setup.py install


EOU
}
vdbi-dir(){ echo $(env-home)/vdbi ; }
vdbi-cd(){  cd $(vdbi-dir); }
vdbi-mate(){ mate $(vdbi-dir) ; }

vdbi-build(){
  rum-
  [ "$(which python)" != "$(rum-dir)/bin/python" ]  && echo $msg ABORT must be inside rumenv to proceed && return 1

  vdbi-install
  vdbi-extras
  vdbi-selinux 

}

vdbi-setup(){
   vdbi-cd
   which python
   python setup.py $*
}

vdbi-install(){ vdbi-setup develop ; }

vdbi-extras(){
  twdev-
  twdev-build
}

vdbi-selinux(){
   apache-
   apache-chcon $(vdbi-dir)
}





