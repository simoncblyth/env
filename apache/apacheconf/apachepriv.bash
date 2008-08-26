
apachepriv-usage(){

  cat << EOU

    CAUTION NOT YET FULLY TESTED 


      apachepriv-url demo      : $(apachepriv-url demo)
               NB the trailing slash is required

      apachepriv-confpath demo : $(apachepriv-confpath demo)

      apachepriv-svnblock <abspath> 
               write the apache fragment to the confpath                                     
               usage example to open up the backup :
                    apachepriv-svnblock $SCM_FOLD/backup
      
      
      apachepriv-block-  <abspath> <name> <users>
               emit to stdout the apache block that provides 
               access to <abspath> under url /<name>/    
               for users specified in the users file
     
  
EOU


}

apachepriv-env(){
  apache-
}


apachepriv-url(){
   echo /$1/
}

apachepriv-confpath(){
  echo $(apache-fragmentpath $1)
}

apachepriv-svnblock(){

   local msg="=== $FUNCNAME :"
   local tmp=/tmp/env/${FUNCNAME/-*/} && mkdir -p $tmp
   local path=$1
   [ -z $path ] && echo $msg ABORT a path is required && return 1
   local name=$(basename $path)    
   local tconf=$tmp/$name.conf

   svn-
   apachepriv-block- $path $name $(svn-userspath) > $tconf
   
   echo $msg temporary conf written to $tconf
   cat $tconf
   
   local conf=$(apachepriv-confpath $name)  
   echo 
   echo $msg proceed to copy this to $conf ? answer Y to proceed 
   
   local ans
   read ans
   if [ "$ans" == "Y" ]; then
      $SUDO cp $tconf $conf 
   fi
   

}


apachepriv-block-(){

  local msg="=== $FUNCNAME :"
 
  local path=$1
  local name=$2
  local users=$3

  [ ! -d $path    ] && echo $msg ERROR no such directory $path && return 1
  [ -z $name      ] && echo $msg ERROR no name supplied           && return 2 
  [ ! -f $users   ] && echo $msg ERROR no such users file $users  && return 3 

  cat << EOL 
#
# created by $BASH_SOURCE  at $(date)
#
#   path    $path
#   name    $name
#   users   $users
#   url     $(apachepriv-url $name)  
#
#
#   this fragment opens path under the  url  to the users specified in the file
#
#

Alias /$name/ "$path/"
<Directory "$path">
    Options Indexes MultiViews
    IndexOptions FancyIndexing NameWidth=*
    AllowOverride None
    
    AuthType Basic
    AuthName "Private Access"
    AuthUserFile "$users"
    Require valid-user
    
    Order allow,deny
    Allow from all
</Directory>


EOL




}