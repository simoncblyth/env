
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
   
      echo =========  temporarily adjusting ownership of $repodir to USER $USER ... will need password
      sudo chown -R $USER:$USER $repodir
      
         
      echo ======== OK proceedinng to load the dumped reposititory commits 
      eval $loadcmd
      
      echo ========= resetting  ownership of $repodir to APACHE2_USER $APACHE2_USER 
      sudo chown -R $APACHE2_USER:$APACHE2_USER $repodir
      
      
   else
      echo OK YOU CHICKENED OUT
   fi      
   
}




svn-add-from-cvs-wc(){

  # following cvs-svn integration recipe from :
  # http://www.cognovis.de/developer/en/subversion_cvs_integration
  #  

  ## construct empty repository 
  
  local name=${name}
  #local repo=http://dayabay.phys.ntu.edu.tw/repos/$name
  local repo=http://localhost/repos/$name
  
  # NEED TO CHECK THAT THE NAME IS AVAILABLE
  # svn mkdir $repo $repo/trunk $repo/branches $repo/tags
  
  svn co $repo/trunk .
  
  #  add the folders from a cvs checkout to the svn repository excluding .svn and CVS folders
  #  (hmm note that CVSROOT folder is not excluded) 
  # 
   find . -type d | grep -v "/CVS$" | grep -v ".svn" | grep -v "^\.$" | sed 's/^\.\//svn add --non-recursive \.\//' | sh

  # we now tell all folders except the current working direoctry to ignore the CVS/ folder
  # so that they are not included in our repository.

  find . -type d | grep -v "/CVS$" | grep -v ".svn" | grep -v "^\.$" | sed 's/^\.\//svn propset svn:ignore *CVS \.\//' | sh



  # Now, that we no add all the files to the repository (except the CVS control files of course)

  find . -type f | grep -v -E "(/CVS|/CVS/Root|/CVS/Repository|/CVS/Entries|/CVS/Entries.Log|/CVS/Entries.Static|/CVS/Tag)$" | grep -v "\.svn\/" | sed 's/^\.\//svn add \.\//' | sh


  svn commit -m "Adding CVS head to $repo/trunk with CVS control files skipped and ignored"


}


svn-dump(){

   local name=${1:-dummy}     #  repository name  
   
   local repodir=$SCM_FOLD/repos/$name
   test -d $repodir || ( echo repodir $repodir not found && return 1 )
   local youngest=$(svnlook youngest $repodir) 
  
   local reva=${2:-0}
   local revb=${3:-$youngest}


   local dir=$SCM_FOLD/svnadmin
   test -d $dir || ( sudo mkdir -p $dir && sudo chown $USER $dir ) 
   
   local label=$name-$reva-$revb   
   cd $dir

   local dumpcmd="svnadmin dump $repodir -r $reva:$revb > $label.dump "
   
   echo $dumpcmd 
   eval $dumpcmd

}





svn-filter-dump(){


   local pabel=$(basename path)
  local blank=""
   local path=${2:-$blank}    #  relative path in the 

   if [ "$pabel" == "/" ]; then
     fabel = $label
   else  
     fabel=$label-$pabel
   fi 
   


 # cannot get this to work, suspect regexps
   #local regexp="branches/$branch"
   #local cmd="svnadmin dump $repodir | $HOME/$SCM_BASE/svndumpfilter2.sh $repodir $regexp  > $label.dump "
   local filtercmd="cat $label.dump | svndumpfilter include branches/$branch  > $fabel.dump 2> $fabel.dump.err  "
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


