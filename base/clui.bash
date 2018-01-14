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

*clui-root*
    Emit repository root folder
    (Mercurial only)

    For SVN would need to shimmy up the directory tree looking 
    for the root .svn folders in 1.6+ OR the last .svn for
    earlier SVN

*clui-path relpath*
    provide repo relative path for a PWD relative path
    (Mercurial only)




EOU
}


clui-cd-with-history(){
  ## use "cd --" for history  "cd -5" to jump  
  . $ENV_HOME/bash/acd_func.sh   
}


clui-chrome(){
   echo "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
}

clui-open(){
   "$(clui-chrome)" ${1:-stackoverflow.com}
}

clui-st()
{
   local msg="$FUNCNAME :"
   local cmd
   if [ -d .hg ]; then 
      cmd="hg status"
   elif [ -d .svn ]; then 
      cmd="svn status"
   elif [ -d .git ]; then 
      cmd="git status"
   else
      cmd="st.py" 
   fi 
   echo $msg $cmd
   eval $cmd
}


clui-alias(){

   alias ss-="sudo su -"
   alias x='exit'
   alias l="ls -lt "
   alias lz="ls -la -Z"
   alias ll="ls -ltra "
   alias lt="ls -lt $se "
   alias pa="ps aux $se"
   alias n="nosetests"
   alias ns="nosetests -s"
   alias nsv="nosetests -s -v"
   alias st="clui-st"
   alias stu="svn st -u"
   alias up="svn up"
   alias sci="svn --username $USER ci "
   alias h='history'
   alias bh='cat ~/.bash_history'
   alias ini='. ~/.bash_profile'
   alias t='type'
   alias f='typeset -F'   ## list functions 
   alias e='cd $ENV_HOME ; hg st '
   alias s='hg st '
   alias vip='vi ~/.bash_profile'
   alias vips='grep BUILD_PATH ~/.bash_profile | grep -v grep '
   alias eu="env-u"
   #alias pyc="find $ENV_HOME -name '*.pyc' -exec rm -f {} \;"
   alias pyc="clui-pyc"
   alias p="clui-path"
   alias pp="clui-ppath"

}

# passing quotes is problematic
#ci(){
#   svn --username $USER ci $*
#}


clui-svnroot-old(){    # shimmy up tree and emit path of last folder containing .svn
   local dir=$1
   local last=$2
   [ "$dir" == "/" ] && echo "" && return
   [ -d "$dir/.svn" ]   && clui-svnroot-old $(dirname $dir) $dir
   [ ! -d "$dir/.svn" ] && echo $last
}

clui-svnroot-new(){    # shimmy up tree and emit path of first folder containing .svn 
   local msg="=== $FUNCNAME :"
   local dir=$1
   #echo $msg $dir 
   [ "$dir" == "/" ] && echo "" && return
   [ ! -d "$dir/.svn" ] && clui-svnroot-new $(dirname $dir)
   [ -d "$dir/.svn" ] && echo $dir
}

clui-svnroot(){  # absolute path of svn repo root folder
   [ "$(which realpath)" == "" ] && echo no realpath : make it with env/tools/realpath/Makefile && return
   local dir=$(realpath ${1:-$PWD})
   local vers=$(svn --version --quiet)
   case $vers in 
      1.6*) clui-svnroot-old $dir ;;
      1.7*) clui-svnroot-new $dir ;;
      1.8*) clui-svnroot-new $dir ;;
   esac
}

clui-gitroot(){ # untested
   git rev-parse --show-toplevel 2>/dev/null 
}

clui-root(){ # Mercurial or SVN root of the 
   local root=$(hg root 2>/dev/null) 
   [ "$root" == "" ] && root=$(clui-svnroot)
   [ "$root" == "" ] && root=$(clui-gitroot)
   echo $root
}

clui-path(){ # normally invoked with alias p, provides repobase/repopath for the relative path argument 
   local path=${1:-$PWD}
   local real=$(realpath $path)
   local root=$(clui-root)
   local base=$(basename $root)
   echo $base/${real/$root\/}
}
clui-ppath(){
   local path=$(clui-path $*)
   echo \`$path\`::
}

clui-pyc(){
   local msg="=== $FUNCNAME : "
   local root=$(clui-root)
   if [ "$root" == "" ]; then
      echo $msg not in hg repo : remove pyc beneath pwd $PWD
      root="."
   else
      echo $msg in hg repo : remove pyc beneath root $root
   fi    
   find $root -name '*.pyc' -exec rm -f {} \;
   find $root -name '*.DS_Store' -exec rm -f {} \;
}

clui-tty(){


   ## this is the bash equivalent of "bindkey -v"

  if [ "$USER" == "blyth" -o "$USER" == "heprez" ]; then
    set -o vi     # vi or emacs CLI editing 
   ##fix delete key operation in vi
    [ -t 0 ] && stty erase '^?'
  fi
  
}  
  
