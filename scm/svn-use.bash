
export SVN_EDITOR="vi"



svn-branch(){

   
   local repo=${1:-dummy}           ## repository name, eg: dyw_release_2_9
   local branch=${2:-dummy}  ## branch name 
   local comment=${3:-dummy}  ## 

   [ "$repo" == "dummy" ]    && echo svn-branch ERROR repo must be specified && return 1 
   [ "$branch" == "dummy" ]  && echo svn-branch ERROR branch must be specified && return 1
   [ "$comment" == "dummy" ] && echo svn-branch ERROR comment  must be specified && return 1

   local base=$SCM_URL/repos/$repo
  
   ##svn copy http://dayabay.ihep.ac.cn/svn/dybsvn/ldm/trunk http://dayabay.ihep.ac.cn/svn/dybsvn/ldm/branches/sjp.issue.234 -m "Branch to resolve issue 234"
   
   local command="svn copy $base/trunk       $base/branches/${USER}-${branch}       -m \"$comment\" " 
   
   echo ===== svn-branch CAUTION, HIGH IMPACT COMMAND ... CHECK CAREFULLY BEFORE COPYING AND RUNNING THE BELOW =====
   echo ===== also reread:  http://svnbook.red-bean.com/nightly/en/svn.branchmerge.using.html 
   echo $command
   
}




svn-load-branch(){

   local name=${1:-dummy}   ## repository name   
   local repodir=$SCM_FOLD/repos/$name
   test -d $repodir || ( echo repodir $repodir not found && return 1 )
   local youngest=$(svnlook youngest $repodir)
   
   local dumpfile=${2:-dummy}
   echo === svn-load-branch from dumpfile $dumpfile into repository at $repodir $youngest ===== 

   local loadcmd="svnadmin load $repodir < $dumpfile "
   echo ====== $loadcmd
   echo ======  DANGER ... ARE YOU SURE YOU WANT TO DO THAT ????  ===== enter YES to proceed
   
   read answer
   if [ "X$answer" == "XYES" ]; then
      echo OK proceedinng
      eval $loadcmd
   else
      echo OK YOU CHICKENED OUT
   fi      
   
}

svn-dump-branch(){

    
   local name=${1:-dummy}     #  repository name  
   
   local repodir=$SCM_FOLD/repos/$name
   test -d $repodir || ( echo repodir $repodir not found && return 1 )
   local youngest=$(svnlook youngest $repodir) 
   
   local branch=${2:-dummy}    #  branch name, eg blyth-optical
   local reva=${3:-0}
   local revb=${4:-$youngest}
   

   local dir=$SCM_FOLD/svnadmin
   test -d $dir || ( sudo mkdir -p $dir && sudo chown $USER $dir ) 
   
   local label=$name-$reva-$revb
   
   
   
   local fabel=$label-$branch   
   cd $dir
   
   # cannot get this to work, suspect regexps
   #local regexp="branches/$branch"
   #local cmd="svnadmin dump $repodir | $HOME/$SCM_BASE/svndumpfilter2.sh $repodir $regexp  > $label.dump "
   local dumpcmd="svnadmin dump $repodir -r $reva:$revb > $label.dump "
   local filtercmd="cat $label.dump | svndumpfilter include branches/$branch  > $fabel.dump 2> $fabel.dump.err  "
   
   echo \"$dumpcmd\"
   eval $dumpcmd

  echo \"$filtercmd\"
   eval $filtercmd



}



svn-use-apache2-add-user(){

   name=${1:-error}
   [ "$name" == "error" ] && echo usage: provide one argument with the username && return 
   [ "$APACHE2_HOME/sbin" == $(dirname $(which htpasswd)) ] || (  echo your PATH to apache2 executables is not setup correctly  && return ) 

   auth="$APACHE2_HOME/$SVN_APACHE2_AUTH"

   echo svn-use-apache2-add-user adding user $name, or potentially changing password if the user exists already

   if [ -f "$auth" ]; then
       htpasswd  -m $auth $name
   else
	   htpasswd -cm $auth $name
   fi

   echo added user to $auth
   cat $auth
}


