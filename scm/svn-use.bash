
export SVN_EDITOR="vi"



svn-branch(){

   
   local repo=${1:-dummy}           ## repository name, eg: dyw_release_2_9
   local fold=${2:-dummy}
   local branch=${3:-dummy}  ## branch name 
   local comment=${4:-dummy}  ## 

   [ "$repo" == "dummy" ]    && echo svn-branch ERROR repo must be specified && return 1 
   [ "$fold" == "dummy" ]    && echo svn-branch ERROR fold must be specified && return 1
   [ "$branch" == "dummy" ]  && echo svn-branch ERROR branch must be specified && return 1
   [ "$comment" == "dummy" ] && echo svn-branch ERROR comment  must be specified && return 1

   local base=http://$SCM_HOST:$SCM_PORT/repos/$repo
  
   ##svn copy http://dayabay.ihep.ac.cn/svn/dybsvn/ldm/trunk http://dayabay.ihep.ac.cn/svn/dybsvn/ldm/branches/sjp.issue.234 -m "Branch to resolve issue 234"
   local command="svn copy $base/trunk/$fold $base/branches/$USER/$fold/$branch -m \"$comment\" "
   
   echo ===== svn-branch CAUTION, HIGH IMPACT COMMAND ... CHECK CAREFULLY BEFORE COPYING AND RUNNING THE BELOW =====
   echo $command
   
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


