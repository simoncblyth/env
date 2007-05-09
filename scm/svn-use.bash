
export SVN_EDITOR="vi"

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


