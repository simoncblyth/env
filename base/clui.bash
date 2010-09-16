clui-src(){ echo base/clui.bash ; }
clui-source(){ echo ${BASH_SOURCE:-$(env-home)/$(clui-src)} ; }
clui-vi(){     vi $(clui-source) ;}

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
   alias l="ls -l "
   alias lz="ls -la -Z"
   alias ll="ls -la "
   alias lt="ls -lt $se "
   alias p="ps aux $se"
   alias st="svn st"
   alias stu="svn st -u"
   alias up="svn up"
   alias ci="svn --username $USER ci "
   alias h='history'
   alias bh='cat ~/.bash_history'
   alias ini='. ~/.bash_profile'
   alias t='type'
   alias f='typeset -F'   ## list functions 
   alias e='cd $ENV_HOME ; svn st '
   alias vip='vi ~/.bash_profile'
   alias vips='grep BUILD_PATH ~/.bash_profile | grep -v grep '
   alias eu="env-u"

}

# passing quotes is problematic
#ci(){
#   svn --username $USER ci $*
#}



clui-tty(){


   ## this is the bash equivalent of "bindkey -v"

  if [ "$USER" == "blyth" ]; then
    set -o vi     # vi or emacs CLI editing 
   ##fix delete key operation in vi
    [ -t 0 ] && stty erase '^?'
  fi
  
}  
  
