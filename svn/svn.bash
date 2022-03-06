svn-source(){ echo ${BASH_SOURCE:-$ENV_HOME/svn/svn.bash} ; }
svn-vi(){        vim $(svn-source) ; }
svn-sourcelink(){ env-sourcelink $(svn-source) ; }
svn-usage(){  cat << EOU
SVN Usage
===============


"svn up" gives a merge conflict because forgot to revert some temporary local changes
----------------------------------------------------------------------------------------

::

    Select: (p) postpone, (df) diff-full, (e) edit, (r) resolved,
            (mc) mine-conflict, (tc) theirs-conflict,
            (s) show all options: p
    C    junoenv-external-libs.sh


1. (p) postpone out of the "svn up"
2. "svn revert junoenv-external-libs.sh" the conflicted discarding local changes
3. now are at trunk with no local changes
4. now can try again "svn up" which should complete cleanly as no local changes


SVN Building
===============


For global settings that are above the details of building and
configuring 

::

     svn-setupdir  : $(svn-setupdir)
     svn-authzpath : $(svn-authzpath)
     svn-userspath : $(svn-userspath)


     svn-authzcheck
         check for duplicated usernames in the authz file


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

     svn-rename <oldname> <newname>
         rename a repo and edits the svn-authzpath file accordingly  
         ... should be done in parallel to the instance using scm-rename          

               
     Precursors...
     
        svnbuild-
        svnsetup-   :  hookup svn and trac with apache   
        svnsync-    :  mirroring setup
        svntools-   :  needed for hot-backup.py on non source svn nodes
        
        
        
    svn-lastrev  <dir1> ... 
        svn-lastrev $SITEROOT/lcgcmt $SITEROOT/../installation/trunk/dybinst    
  

    == CMDS FOR "WC AS SOURCE" DEGRADED USAGE ==

       Workaround during times of server troubles ... when it is prudent not
       to commit into the repository.   Designate the repository 
       working copy on a particular node as '''THE ONE''' source and propagate
       changes made to this WC via patch transfers over scp.

       Use consistent WC folder naming convention ... to remind you and to act as
       input to these commands, eg for node P as the source working copy in folder "P:env.P"
       name WC folders on "child" nodes to "env.P"   

       svn-hometag   : $(svn-hometag)      tag of source WC, assuming directory naming convention
             U means that not following convention ... meaning not in degraded usage mode
       svn-patchtag  \$ENV_HOME : $(svn-patchtag $ENV_HOME) 
       svn-patchname            : $(svn-patchname)     name for patch files 
       svn-patchloc             : $(svn-patchloc)     scp coordinates of the patch file 

       svn-makepatch    
            creates a patch in \$ENV_HOME with standard name


       svn-getpatch 
            scp the patch file to local WC

       svn-revert 
            recursive revert the \$ENV_HOME working copy to a pristine state 
            iff a corresponding patch file with the correct revision exists

       svn-applypatch
            apply the patch to the \$ENV_HOME working copy  
     
       svn-checkpatch
            compare the result of "svn diff" with the patch file ... to verify that the patching 
            worked correctly


       svn-autopatch 
            do all 4 prior steps 

  Set up "child" working copy, naming convention defines '''THE SOURCE''' :
      G>   cd \$HOME
      G>   svn checkout \$(env-url) env.P     ## using node P WC at ssh location P:env.P

  After developments on node P 
      P>  svn-makepatch 

  Propagate to "child" WC such as G with :
      G>   svn-getpatch         
      G>   svn-revert          
      G>   svn-applypatch      
      G>   svn-checkpatch

   OR 
      G>  svn-autopatch     
      
   ISSUES 
         Do patches fully capture deletion and additions correctly ??




PRE-COMMIT HOOK IDEAS
-----------------------

Disallowing tabs in python files

#. http://wordaligned.org/articles/a-subversion-pre-commit-hook

Ignoring Directories
----------------------

::

    delta:workflow blyth$ svn propedit svn:ignore .
    Set new value for property 'svn:ignore' on '.'

    ## added the translated wiki, ticket directories to the ignores list 
    ## add not yet ready to add to repo, but need to test integrated sphinx 
    ## builds of full docs




                         
EOU

}


svn-revert-(){ svn --recursive revert . ; }

svn-log5(){ svn log --limit ${1:-5} -v ; }
svn-addx(){ $FUNCNAME- $* ; }
svn-addx-(){ local name=$1 ; cat << EOC
svn add $name ; svn ps svn:executable yes $name
EOC
}

svn-alias(){
  alias log5="svn-log5"
  alias addx="svn-addx"
}


svn-hometag(){   
   local tag=${ENV_HOME/*./}
   [ "$ENV_HOME" == "$tag" ] && echo U || echo $tag 
}
svn-patchtag(){  echo $(svn-lastrev- $ENV_HOME) ; }
svn-patchname(){ echo $(svn-patchtag).patch ; }
svn-patchpath(){ echo $ENV_HOME/$(svn-patchname) ; }
svn-patchloc(){  echo $(svn-hometag):$(basename $ENV_HOME)/$(svn-patchname) ; }

svn-ispristine-(){
  local dir=$1
  local lastrev=$(svn-lastrev $dir)
  local version=$(svnversion $dir)
  [ "$lastrev" == "$version" ] && return 0 || return 1
}

svn-revert(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   [ ! -f "$(svn-patchpath)" ] && echo $msg ABORT patch file $(svn-patchpath) does not exist cannot revert $PWD && return 1 
   cd $ENV_HOME
   svn-revert-
   cd $iwd
}

svn-makepatch(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   cd $ENV_HOME
   local name=$(svn-patchname)
   [ "X$1" != "X" ] && name="$name.$1"
   echo $msg creating patch $name in $PWD 
   svn diff . > $name
   cd $iwd
}

svn-getpatch(){
  local msg="=== $FUNCNAME :"

  local tag=$(svn-hometag)
  [ "$tag" == "" ]          && echo $msg svn-hometag indicates ENV_HOME : $ENV_HOME is not using the degraded convention to identify source WC eg : env.P ... ABORTING && return 1
  [ "$tag" == "$NODE_TAG" ] && echo $msg fromtag $tag must NOT be same as current tag $NODE_TAG && return 1
  [ ! -d "$ENV_HOME" ]      && echo $msg ENV_HOME $ENV_HOME does not exist ... ABORT && return 1 

  local iwd=$PWD
  cd $ENV_HOME
  local cmd="scp $(svn-patchloc) . "
  local ans
  read  -p " $msg from $PWD perform: $cmd  , enter YES to proceed " ans
  if [ "$ans" == "YES" ]; then
     echo $msg OK, PROCEEDING 
     $cmd
  else
     echo $msg OK, SKIPPING
  fi
  cd $iwd 
}

svn-applypatch(){
   local msg="=== $FUNCNAME :"

   local tag=$(svn-hometag)
   [ "$tag" == "" ]          && echo $msg svn-hometag indicates ENV_HOME : $ENV_HOME is not using the degraded convention to identify source WC eg : env.P ... ABORTING && return 1
   [ "$tag" == "$NODE_TAG" ] && echo $msg fromtag $tag must NOT be same as current tag $NODE_TAG && return 1
   [ ! -d "$ENV_HOME" ]      && echo $msg ENV_HOME $ENV_HOME does not exist ... ABORT && return 1 
   
   local iwd=$PWD
   cd $ENV_HOME

   ! svn-ispristine- $PWD    && echo $msg ABORT $PWD is not pristine working copy ... consider doing recursive revert eg with : svn-revert    && return 1
   local name=$(svn-patchname)   
   [ ! -f "$name" ]          && echo $msg ABORT no patch file exists with name $name : try  svn-getpatch  && return 1

   local cmd="patch -p0 < $name"
   local ans
   read -p "$msg proceed with patch command : $cmd ... enter YES to proceed "  ans

   if [ "$ans" == "YES" ]; then
     echo $msg proceeding ...
     eval $cmd
   else
     echo $msg skipping...
   fi

}

svn-checkpatch(){
    local msg="=== $FUNCNAME :"
    local name=$(svn-patchname)   
    svn-ispristine- $PWD && echo $msg ABORT $PWD IS PRISTINE working copy ... nothing to compare   && return 1
    [ ! -f "$name" ]     && echo $msg ABORT no patch file exists with name $name && return 1
    svn-makepatch- check 
    [ ! -f "$name.check" ] && echo $msg ABORT no patch check file exists with name $name.check && return 1
    echo $msg compare the patch file $name that was applied with $name.check    

    local cmd="diff $name $name.check"
    echo $msg cmd $cmd
    eval $cmd
}


svn-autopatch(){

   svn-getpatch
   svn-revert 
   svn-applypatch
   svn-checkpatch

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
svntools-(){      . $ENV_HOME/svn/svntools.bash            && svntools-env  $* ; }

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


svn-authzcheck(){
   local msg="=== $FUNCNAME :"
   echo $msg looking for duplicates in $(svn-authzpath)
   perl -n -e 'm,^developers = (.*)$, && print split(" ",$1) '  $(svn-authzpath)  | tr "," "\n" | sort | uniq -D
}



svn-mode(){ echo ${SVN_MODE:-$(svn-mode-default $*)} ; }
svn-mode-default(){
  case ${1:-$NODE_TAG} in
Y2|Y1|ZZ|C|AA|HKU) echo system ;;
                G) echo systemport ;;
                *) echo source ;;
  esac
}



svn-ver(){
  case ${1:-$NODE_TAG} in 
    C) echo 1.4.2 ;;
C2|C2R) echo 1.4.6 ;;
   XX) echo 1.4.3 ;; 
   YY) echo 1.4.3 ;;
    *) echo 1.4.0 ;;
  esac
}

svn-env(){

  elocal-
  apache- 

  [ "$NODE_APPROACH" == "stock" ] && return 0
  [ "$(svn-mode)" == "system" ]   && return 0

  local ver=$(svn-ver)
  export SVN_NAME=subversion-$ver
  export SVN_NAME2=subversion-deps-$ver  
    
  #export SVN_BUILD=$SYSTEM_BASE/svn/build/$SVN_NAME
  export SVN_HOME=$SYSTEM_BASE/svn/$SVN_NAME
  
  export PYTHON_PATH=$SVN_HOME/lib/svn-python:$PYTHON_PATH
  
  svn-path
}

svn--(){
   sudo bash -c "export ENV_HOME=$ENV_HOME ; . $ENV_HOME/env.bash ; svn- ; $* "
}

svn-hotbackuppath-system(){
   svntools-
   case ${1:-$NODE_TAG} in
AA|HKU|Y1|Y2) echo /usr/share/doc/subversion-1.6.11/tools/backup/hot-backup.py ;;
           *) echo $(svntools-dir)/tools/backup/hot-backup.py  ;;  ## as stock svn doesnt come with the tools
   esac
}

svn-hotbackuppath(){
  local mode=$(svn-mode)
  if [ "${mode:0:6}" == "system" ]; then
      svn-hotbackuppath-system     
  elif [ "$mode" == "source" ]; then  
      svnbuild-
      echo $(svnbuild-dir)/tools/backup/hot-backup.py    
  fi
}

svn-dumpload-incremental(){
   case ${1:0:1} in
      0) echo -n ;;
      *) echo --incremental ;;
   esac 
}


svn-dumpload-rngs(){ 
  echo "0:1000 1001:2000 2001:3000 3001:4000 4001:5000 5001:5902 5904:5934"
}

svn-dumpload(){

   local msg="=== $FUNCNAME :"
   local repo=svn/dybsvn

   local iwd=$PWD 
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp
   cat << EOC
       mkdir -p $tmp
       cd $tmp 
EOC

   local rng
   local inc
   for rng in $(svn-dumpload-rngs) ; do
       cat << EOC
       echo $msg \$(date)  dumping $rng
       svnadmin dump `local-scm-fold`/$repo --revision $rng $(svn-dumpload-incremental $rng) > $rng.txt
EOC
   done 
   cat << EOC
       mkdir -p $(dirname $tmp/$repo)
       svnadmin create $tmp/$repo
EOC

   for rng in $(svn-dumpload-rngs) ; do 
       cat << EOC
       echo $msg \$(date)  loading $rng
       svnadmin load $tmp/$repo < $rng.txt
EOC
   done
   #cd $iwd
  

}


svn-pdumpload(){

   local msg="=== $FUNCNAME :"
   local repo=svn/dybsvn

   local iwd=$PWD 
   local tmp=/tmp/env/$FUNCNAME && mkdir -p $tmp

   cat << EOC
       mkdir -p $(dirname $tmp/$repo)
       svnadmin create $tmp/$repo
EOC

   local rng
   for rng in $(svn-dumpload-rngs) ; do
       cat << EOC
       echo $msg \$(date)  dumping and loading $rng
       svnadmin dump `local-scm-fold`/$repo --revision $rng $(svn-dumpload-incremental $rng) | svnadmin load $tmp/$repo
EOC
   done 

   #cd $iwd
  
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

svn-repo-dirname-forsite(){
   case ${1:-ntu} in 
        ihep) echo svn ;;
         ntu) echo repos ;;
           *) echo repos ;;
   esac 
}

svn-repo-path(){
   local name=${1:-dummy}
   trac-
   local site=$(trac-site $name)
   echo $SCM_FOLD/$(svn-repo-dirname-forsite $site)/$name
}

svn-postcommit-path(){
   local name=${1:-dummy}
   echo $(svn-repo-path $name)/hooks/post-commit
}

svn-repos(){
   local site=${1:-ntu}
   local iwd=$PWD
   cd $SCM_FOLD/$(svn-repo-dirname-forsite $site)
   for name in $(ls -1)
   do
      [ -d $name ] && echo $name
   done
   cd $iwd
}

svn-exists(){
   local name=$1
   trac-
   local site=$(trac-site $name)
   local repo
   for repo in $(svn-repos $site) ; do
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

    svn-- svn-create- $name $arg

    cd $iwd
}


svn-create-(){

    local msg="=== $FUNCNAME :"
    local name=$1
    local arg=$2
    local repo=$(svn-repo-path $name)
    local dir=$(dirname $repo) 
    
    [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p "$dir"
    cd $dir
    
    local cmd="svnadmin create $name"
    [ ! -d $name ] && echo $msg $cmd && eval $cmd

    local pop=0
    case $arg in 
      EMPTY) echo $msg leaving empty repository ;;
       INIT) pop=1 ; svn-populate       ;;
          *) pop=1 ; svn-populate $arg  ;;     
    esac  
       
    if [ "$pop" == "1" ] ; then
       local tmp=$(svn-tmpdir)   
       local imd="svn import $tmp file://$repo -m \"initial import by $(svn-sourcelink) '''$FUNCNAME''' on $(date) with argument $arg \" "
       echo $msg $imd
       eval $imd
    fi
    
    svn-chown $name
    

}

svn-chown(){
    local msg="=== $FUNCNAME :"
    local name=$1
    [ -z "$name" ] && echo $msg the name of an svn repository is a required argument && return 1  
    local repo=$(svn-repo-path $name)
    local user=$(apache- ; apache-user)
    local cmd="$SUDO chown -R $user:$user $repo/"
    echo $msg $cmd
    eval $cmd
}


svn-tmpdir(){ echo /tmp/env/$FUNCNAME/$$  ; }

svn-populate(){
   local dir=$1
   local tmp=$(svn-tmpdir)  && mkdir -p $tmp/{branches,tags,trunk}
   [ -n "$dir" -a -d "$dir" ] && cp -r $dir $tmp/trunk/  || echo $msg starting with just branches/tags/trunk 
}


svn-rename(){

   local iwd=$PWD
   local msg="=== $FUNCNAME :"
   local tmpd=/tmp/env/$FUNCNAME && mkdir -p $tmpd
   local oldname=$1
   local newname=$2
   [ -z $SCM_FOLD ] && echo $msg ABORT no SCM_FOLD && return 1
 
   ! svn-exists $oldname && echo $msg ABORT no such repository exists with name \"$oldname\" && return 1
   svn-exists $newname   && echo $msg ABORT a repository exists already with name \"$newname\" && return 1
    
   local oldrepo=$(svn-repo-path $oldname)
   local dir=$(dirname $oldrepo)
   cd $dir

   local cmd="sudo mv $oldname $newname"
   echo $msg $cmd
   eval $cmd

   local authz=$(svn-authzpath)
   local tmpz=$tmpd/$(basename $authz)

   echo $msg CAUTION THIS IS EDITING A DERIVED FILE USE svnsetup- authz- authz-vi to look at the generator  

   perl -p -e "s,$oldname:,$newname:,"  $authz > $tmpz
   diff $authz $tmpz
   local zmd="cp $tmpz $authz "
   local ans
   read -p "$msg proposes to change the authz file $authz ... with $zmd ... , to proceed enter YES " ans
   if [ "$ans" == "YES" ]; then
      echo $msg proceeding
      eval $zmd
   else
      echo $msg skipping 
   fi 

   cd $iwd
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

svn-uuid-(){
  svn info $1 | env -i perl -n -e 'm/^Repository UUID: (\S*)$/ && print "$1\n" '
}



svn-change-password(){
   local msg=
   local user=$1
   local pass=$2

   [ -z "$user" ] && echo $msg a username is required && return 1
   [ -z "$pass" ] && echo $msg a new password is required && return 1

   local path=$(svn-userspath)
   local tmp=/tmp/env/$USER/$FUNCNAME/$(basename $path)
   mkdir -p $(dirname $tmp)
   cp $path $tmp
   cp $path $tmp.safetycopy

   ls -l $path
   ls -l $tmp*
   $(env-home)/trac/auth/htpasswd.py -b $tmp $user $pass
 
   local cmd="diff $path $tmp"
   echo $msg $cmd  slated changes 
   eval $cmd
   echo 

   echo $msg propagate the above change into the users file with below copy 
   cmd="sudo cp $tmp $path"
   echo $msg $cmd
   local answer
   read -p "$msg Enter YES to proceed : " answer
   if [ "$answer" == "YES" ]; then 
         echo $msg : proceeding with \"$cmd\"
         echo $msg : sudoer password may be needed
         eval $cmd
   else
         echo $msg : OK skipping 
   fi 
   
   cmd="rm $tmp $tmp.safetycopy"
   echo $msg tidying temporaries \"$cmd\"
   eval $cmd

   cmd="ls -l $path"
   echo $msg checking users file \"$cmd\"
   eval $cmd 

}



svn-offline-blyth(){
   cd ~/junotop/offline
   svn log -v --search blyth > ~/$FUNCNAME.log
}


