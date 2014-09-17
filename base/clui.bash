clui-src(){ echo base/clui.bash ; }
clui-source(){ echo ${BASH_SOURCE:-$(env-home)/$(clui-src)} ; }
clui-vi(){     vi $(clui-source) ;}

clui-env(){
   
   local dbg=${1:-0}
   local msg="=== $FUNCNAME :"
   
   [ "$dbg" == "1" ] && echo $msg
   
   export EDITOR=vi
   
   clui-alias
   clui-cd-with-history
   clui-tty 
    
}

clui-usage(){ cat << EOU

EOU
}


clui-cd-with-history(){
  ## use "cd --" for history  "cd -5" to jump  
  . $ENV_HOME/bash/acd_func.sh   
}

clui-alias(){

   alias ss-="sudo su -"
   alias x='exit'
   alias l="ls -lt "
   alias lz="ls -la -Z"
   alias ll="ls -ltra "
   alias lt="ls -lt $se "
   alias p="ps aux $se"
   alias n="nosetests"
   alias ns="nosetests -s"
   alias nsv="nosetests -s -v"
   alias st="hg st"
   alias sst="svn st"
   alias stu="svn st -u"
   alias up="svn up"
   alias sci="svn --username $USER ci "
   alias h='history'
   alias bh='cat ~/.bash_history'
   alias ini='. ~/.bash_profile'
   alias t='type'
   alias f='typeset -F'   ## list functions 
   alias e='cd $ENV_HOME ; hg st '
   alias vip='vi ~/.bash_profile'
   alias vips='grep BUILD_PATH ~/.bash_profile | grep -v grep '
   alias eu="env-u"
   #alias pyc="find $ENV_HOME -name '*.pyc' -exec rm -f {} \;"
   alias pyc="clui-pyc"

}

# passing quotes is problematic
#ci(){
#   svn --username $USER ci $*
#}

clui-pyc(){
   local msg="=== $FUNCNAME : "
   local root=$(hg root 2>/dev/null)
   if [ "$root" == "" ]; then
      echo $msg not in hg repo : remove pyc beneath pwd $PWD
      root="."
   else
      echo $msg in hg repo : remove pyc beneath root $root
   fi    
   find $root -name '*.pyc' -exec rm -f {} \;
}

clui-tty(){


   ## this is the bash equivalent of "bindkey -v"

  if [ "$USER" == "blyth" -o "$USER" == "heprez" ]; then
    set -o vi     # vi or emacs CLI editing 
   ##fix delete key operation in vi
    [ -t 0 ] && stty erase '^?'
  fi
  
}  
  
