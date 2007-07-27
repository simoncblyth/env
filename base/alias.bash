

[ "$BASE_DBG" == "1" ] && echo alias.bash

alias-x(){ scp $HOME/$BASE_BASE/alias.bash ${1:-$TARGET_TAG}:$BASE_BASE ; }


alias x='exit'
alias l='ls -l '
alias ll='ls -la '
alias lt="ls -lt"
alias h='history'
alias bh='cat ~/.bash_history'
alias ini='. ~/.bash_profile'


alias cq="condor_q"

alias vr='vi `find . -name requirements`'

alias-chkdep(){
	perl -n -e '@a=split / /, $_; printf "$_\n" for(@a) ;' $1 
}

