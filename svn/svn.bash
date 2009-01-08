svn-source(){ echo $BASH_SOURCE ; }
svn-sourcelink(){ env-sourcelink $(svn-source) ; }
svn-usage(){
  
   cat << EOU

    For global settings that are above the details of building and
    configuring 

     svn-setupdir  : $(svn-setupdir)
     svn-authzpath : $(svn-authzpath)
     svn-userspath : $(svn-userspath)


     svn-global-ignores :  
         handholding only
     
     svn-create  <name>  <arg>
     
          create a repository called <name> in a standard location $SCM_FOLD/repos/<name>
          and import the content of <arg> if it is a directory path into trunk, 
          if arg is "EMPTY" then leave the repository at revision zero, otherwise just
          create the branches/tags/trunk structure 
                 
          issues for apache presentation ... the repository should be owned by APACHE_USER       
                        
     svn-wipe <name>
     
          delete the repository called <name>
                         
     Precursors...
     
        svnbuild-
        svnsetup-   :  hookup svn and trac with apache   
        svnsync-    :  mirroring setup
        
        
        
    svn-lastrev  <dir1> ... 
        svn-lastrev $SITEROOT/lcgcmt $SITEROOT/../installation/trunk/dybinst    
        
                         
EOU

}


#
#  these precursors and content are mostly deprecated 
#
# swig-(){         . $ENV_HOME/svn/swig.bash      && swig-env $* ; } 
# svn-apache2-(){  . $ENV_HOME/svn/svn-apache2.bash && svn-apache2-env $* ; }
# svn-sync-(){     . $ENV_HOME/svn/svn-sync.bash  && svn-sync-env  $* ; } 
# svn-tools-(){    . $ENV_HOME/svn/svn-tools.bash && svn-tools-env $* ; }
# svn-build-(){    . $ENV_HOME/svn/svn-build.bash && svn-build-env $* ; } 
# svn-tmp-(){      . $ENV_HOME/svn/svn-tmp.bash   && svn-tmp-env   $* ; } 
#


svnbuild-(){      . $ENV_HOME/svn/svnbuild/svnbuild.bash   && svnbuild-env $* ; } 
svnsetup-(){      . $ENV_HOME/svn/svnconf/svnsetup.bash    && svnsetup-env $* ; }
svnsync-(){       . $ENV_HOME/svn/svnsync/svnsync.bash     && svnsync-env  $* ; }

svn-setupdir(){
  case ${1:-$NODE_TAG} in 
     H) echo  $(apache-confdir)          ;;
 old-G) echo  $(apache-confdir)/local    ;;
     *) echo  $(apache-confdir)/svnsetup ;;
  esac
}

svn-authzname(){ 
  case ${1:-$NODE_TAG} in
 H|old-G)  echo svn-apache2-authzaccess ;;
       *)  echo authz.conf              ;;
  esac    
}

svn-usersname(){ 
  case ${1:-$NODE_TAG} in
 H|old-G) echo svn-apache2-auth ;;
       *) echo users.conf       ;; 
  esac   
}

svn-authzpath(){ echo $(svn-setupdir $*)/$(svn-authzname $*) ; }
svn-userspath(){ echo $(svn-setupdir $*)/$(svn-usersname $*) ; }


svn-env(){

  elocal-
  apache- 

  [ "$NODE_APPROACH" == "stock" ] && return 0

  local ver
  case $NODE_TAG in 
    C) ver=1.4.2 ;;
   XX) ver=1.4.3 ;; 
    *) ver=1.4.0 ;;
  esac
  
  export SVN_NAME=subversion-$ver
  export SVN_NAME2=subversion-deps-$ver  
    
  #export SVN_BUILD=$SYSTEM_BASE/svn/build/$SVN_NAME
  export SVN_HOME=$SYSTEM_BASE/svn/$SVN_NAME
  
  export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH
  
  svn-path
}




svn-hotbackuppath(){
  svnbuild-
  case ${1:-$NODE_TAG} in
     G) echo $(local-base)/svn/tools/backup/hot-backup.py  ;;  ## as stock svn doesnt come with the tools
     *) echo $(svnbuild-dir)/tools/backup/hot-backup.py    ;;
  esac
}




svn-path(){

  local dirs="$SVN_HOME/lib/svn-python/svn $SVN_HOME/lib/svn-python/libsvn"
  for dir in $dirs
  do 
     env-llp-prepend $dir
  done
  

  env-prepend $SVN_HOME/bin
  
}


svn-global-ignores(){

cat << EOI
#  uncomment global-ignores in [miscellany] section of
#     $HOME/.subversion/config
#  setting it to : 
#
global-ignores = setup.sh setup.csh cleanup.sh cleanup.csh Linux-i686* Darwin* InstallArea load.C
#
#    NB there is no whitespace before "global-ignores"
# 
#  after this   
#        svn status -u 
#  should give a short enough report to be useful
#
EOI

echo vi $HOME/.subversion/config


}


svn-repo-dirname(){
  case ${1:-$NODE_TAG} in
  XX|XT) echo svn ;;
      *) echo repos ;;
  esac    
}


svn-repo-path(){
   local name=${1:-dummy}
   echo $SCM_FOLD/$(svn-repo-dirname)/$name
}


svn-repos(){
   local iwd=$PWD
   cd $SCM_FOLD/$(svn-repo-dirname)
   for name in $(ls -1)
   do
      [ -d $name ] && echo $name
   done
   cd $iwd
}

svn-exists(){
   local name=$1
   local repo
   for repo in $(svn-repos) ; do
      [ "$name" == "$repo" ] && return 0
   done
   return 1
}


svn-create(){
    local iwd=$PWD
    local msg="=== $FUNCNAME :"
    local name=${1:-dummy}  
    
    [ -z "$name" ]     && echo $msg an instance name must be provided && return 1
    svn-exists $name   && echo $msg ABORT a repository with name \"$name\" exists already && return  1 
    
    local arg=${2:-EMPTY}  
                              
    [ -z $SCM_FOLD ] && echo $msg ABORT no SCM_FOLD && return 1

    local repo=$(svn-repo-path $name)
    local dir=$(dirname $repo) 
    
    [ ! -d "$dir" ] && echo $msg creating dir $dir && $SUDO mkdir -p "$dir"
    cd $dir
    
    local cmd="$SUDO svnadmin create $name"
    [ ! -d $name ] && echo $msg $cmd && eval $cmd

    case $arg in 
      EMPTY) echo $msg leaving empty repository ;;
       INIT) svn-populate       ;;
          *) svn-populate $arg  ;;     
    esac  
       
    local tmp=$(svn-tmpdir)   
    local imd="$SUDO svn import $tmp file://$repo -m \"initial import by $(svn-sourcelink) '''$FUNCNAME''' on $(date) with argument $arg \" "
    echo $msg $imd
    eval $imd
       
    cd $iwd
}

svn-tmpdir(){ echo /tmp/env/$FUNCNAME/$$  ; }

svn-populate(){
   local dir=$1
   local tmp=$(svn-tmpdir)  && mkdir -p $tmp/{branches,tags,trunk}
   [ -n "$dir" -a -d "$dir" ] && cp -r $dir $tmp/trunk/  || echo $msg starting with just branches/tags/trunk 
}


svn-wipe(){

   local iwd=$PWD
   local msg="=== $FUNCNAME :"
   local name=$1
   [ -z $SCM_FOLD ] && echo $msg ABORT no SCM_FOLD && return 1
   
   ! svn-exists $name && echo $msg ABORT no such repository exists with name \"$name\" && return 1
   
   local repo=$(svn-repo-path $name)
   local dir=$(dirname $repo) 
   
   cd $dir
   [ ! -d $name ] && echo $msg ABORT repo $name does not exist && return 1
   
   local answer
   read -p "$msg are you sure you want to wipe the repository \"$name\" from $dir ? YES to proceed " answer
   [ "$answer" != "YES" ] && echo $msg skipping && return 1
   [ ${#name} -lt 3 ]  && echo $msg name $name is too short not proceeding && return 1
   
   local cmd="$SUDO rm -rf \"$name\""
   echo $msg $cmd 
   eval $cmd

   cd $iwd
}



svn-lastrev-(){
  svn info $1 | env -i perl -n -e 'm/^Last Changed Rev: (\d*)$/ && print "$1\n" '
}

svn-lastrev(){
  while [ $# -gt 0 ] ; do
     svn-lastrev- $1
     shift 1
  done 
}



