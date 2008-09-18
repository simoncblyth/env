

clui-env(){
   
   local dbg=${1:-0}
   local msg="=== $FUNCNAME :"
   
   [ "$dbg" == "1" ] && echo $msg
   
   export EDITOR=vi
   
   clui-alias
   clui-tty
    
}


clui-alias(){

   alias x='exit'
   alias l='ls -l '
   alias ll='ls -la '
   alias lt="ls -lt"
   alias st="svn st"
   alias stu="svn st -u"
   alias up="svn up"
   alias ci="svn ci"
   alias h='history'
   alias bh='cat ~/.bash_history'
   alias ini='. ~/.bash_profile'
   alias t='type'
   alias f='typeset -F'   ## list functions 
   alias e='cd $ENV_HOME'

}


clui-tty(){

   ##fix delete key operation in vi
  [ -t 0 ] && stty erase '^?'

   ## this is the bash equivalent of "bindkey -v"

  if [ "$USER" == "blyth" ]; then
    set -o vi     # vi or emacs CLI editing 
  fi
  
}  
  