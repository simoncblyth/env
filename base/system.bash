system-usage(){ cat << EOU

EOU
}

system-log(){

   local msg="=== $FUNCNAME :"
   
   ## [ "$(uname)" == "Linux" ] && echo $msg is a Darwin thang && return 1 
   ## actually tis on Linux also , but not investigated "man logger"
   
   local def="dummy entry from $msg  "
   local tag=${1:-$msg}
   local ent=${2:-$def}
   
   # sends message to system.log, tagged by the name of the script
   /usr/bin/logger -i -p daemon.notice -t $tag $ent

   # use Console.app for convenient access to the system.log  on darwin

}
