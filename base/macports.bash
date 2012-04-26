# === func-gen- : base/macports fgp base/macports.bash fgn macports fgh base
macports-src(){      echo base/macports.bash ; }
macports-source(){   echo ${BASH_SOURCE:-$(env-home)/$(macports-src)} ; }
macports-vi(){       vi $(macports-source) ; }
macports-usage(){
  cat << EOU
     macports-src : $(macports-src)
     macports-dir : $(macports-dir)



   COMMANDS

       port installed
       port list installed     ## not the same as above, and much slower 



#######################################################################################################
#
#  to work with system /usr/bin/python (eg for trac) formerly commented  
#  macports hookup in ~/.bash_profile,  but that is painful as 
#     svn: This client is too old to work with working copy '.'; please get a newer Subversion client
#
#  forcing kludge aliasing:
#
#      #local bin=/opt/local/bin
#      #alias svn="$bin/svn"
#      #alias svnversion="$bin/svnversion"
#
#  try using "port select"  see "port help select"
#  
#  port select --list python 
#  Available versions for python:
#        none
#        python25
#        python25-apple
#        python26 (active)
#
#  sudo port select python python25         ##  python 2.5.5
#  sudo port select python python25-apple   ##  python 2.5.1  still /opt/local/bin/python
#  sudo port select python none             ##  python 2.5.1  direct /usr/bin/python
#
#  port select --show python 
#
#########################################################################################################

EOU
}
macports-dir(){ echo $(local-base)/env/base/base-macports ; }
macports-cd(){  cd $(macports-dir); }
macports-mate(){ mate $(macports-dir) ; }
macports-get(){
   local dir=$(dirname $(macports-dir)) &&  mkdir -p $dir && cd $dir
}

macports-env(){
   elocal- 
   ## avoid stomping on the virtualenv
   if [ -z "$VIRTUAL_ENV" ]; then
       export PATH=/opt/local/bin:/opt/local/sbin:$PATH
       export MANPATH=/opt/local/share/man:$MANPATH
   fi
}


