# === func-gen- : hudson/hudson fgp hudson/hudson.bash fgn hudson fgh hudson
hudson-src(){      echo hudson/hudson.bash ; }
hudson-source(){   echo ${BASH_SOURCE:-$(env-home)/$(hudson-src)} ; }
hudson-vi(){       vi $(hudson-source) ; }
hudson-env(){      elocal- ; }
hudson-usage(){
  cat << EOU
     hudson-src : $(hudson-src)
     hudson-dir : $(hudson-dir)

     http://wiki.hudson-ci.org/display/HUDSON/Meet+Hudson


    http://hudson-labs.org/


  == hudson-redmine plugin ! ==

     http://wiki.hudson-ci.org/display/HUDSON/Redmine+Plugin
         adds redmine ticket links to the hudson scm changes 

     https://github.com/HudsonLabs/redmine-plugin
     https://github.com/hudson/redmine-plugin


EOU
}
hudson-dir(){ echo $(local-base)/env/hudson/hudson-hudson ; }
hudson-cd(){  cd $(hudson-dir); }
hudson-mate(){ mate $(hudson-dir) ; }
hudson-get(){
   local dir=$(dirname $(hudson-dir)) &&  mkdir -p $dir && cd $dir

}
