# === func-gen- : rum/rumdev fgp rum/rumdev.bash fgn rumdev fgh rum
rumdev-src(){      echo rum/rumdev.bash ; }
rumdev-source(){   echo ${BASH_SOURCE:-$(env-home)/$(rumdev-src)} ; }
rumdev-vi(){       vi $(rumdev-source) ; }
rumdev-env(){      elocal- ; }
rumdev-usage(){
  cat << EOU
     rumdev-src : $(rumdev-src)
     rumdev-dir : $(rumdev-dir)



    tw.rum-exp  no changes since Jan 2009 


EOU
}
rumdev-dir(){ echo $(local-base)/env/rumdev ; }
rumdev-cd(){  cd $(rumdev-dir); }
rumdev-mate(){ mate $(rumdev-dir) ; }

rumdev-rbase(){  echo http://hg.python-rum.org ; }
rumdev-repos(){  echo RumAlchemy RumDemo RumSecurity TgRum TgRumDemo WebFlash rum tw.rum tw.rum-exp ; } 
rumdev-get(){
   local msg="=== $FUNCNAME :"
   local dir=$(rumdev-dir) &&  mkdir -p $dir && cd $dir
   local repo ; for repo in $(rumdev-repos) ; do
       [ ! -d "$repo" ] && hg clone $(rumdev-rbase)/$repo || echo $msg repo $repo  is already cloned
   done
}

rumdev-diff-(){
   python-
   local repo=${1:-rum}
   local pkgn=${2:-$(rumdev-repo2pkgn $repo)}
   local mdir=$(python-mdir $pkgn 2>/dev/null)

   [ "$mdir" == "" ] && echo $msg skip pkgn $pkgn as no mdir && return 1

   local eggd=$(dirname $mdir)
   local devd=$(rumdev-dir)/$pkgn
   local reld=$(rumdev-repo2rel $repo)
   [ -n "$reld" ] && devd="$devd/$reld"

   echo eggd $eggd devd $devd
   #ls -l $eggd
   #ls -l $devd

   if [ -d "$eggd" -a -d "$devd" ] ; then
      local cmd="diff -r --brief $eggd $devd | grep -v .pyc | grep -v .hg  "
      echo $cmd
      eval $cmd
   else
      [ ! -d "$eggd" ] && echo missing eggd $eggd  
      [ ! -d "$devd" ] && echo missing devd $devd  
   fi
 
}


rumdev-lower(){
     tr "[A-Z]" "[a-z]" 
}

rumdev-repo2pkgn(){
   local repo=$1
   case $repo in 
     RumAlchemy|RumDemo|RumSecurity|TgRum|TgRumDemo|WebFlash) echo $repo | rumdev-lower  ;;
               *) echo $repo ;; 
   esac
}

rumdev-repo2rel(){
   local repo=$1
   case $repo in 
       tw.rum) echo tw ;;
   esac
}


rumdev-diff(){
   rumdev-cd  
   local repo ; for repo in $(rumdev-repos) ; do
      rumdev-diff- $repo
   done
}


rumdev-install(){

   local msg="=== $FUNCNAME :"
   rum-
   [ "$(which python)" != "$(rum-dir)/bin/python" ] && echo $msg ABORT this must be run whilst inside the rum virtualenv  && return 1 


   local tips="tw.rum rum"

   local tip ; for tip in $tips ; do
      rumdev-cd
      cd $tip
      python setup.py develop
   done 
 


}



