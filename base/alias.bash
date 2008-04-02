

[ "$BASE_DBG" == "1" ] && echo alias.bash

alias-x(){ scp $HOME/$BASE_BASE/alias.bash ${1:-$TARGET_TAG}:$BASE_BASE ; }



alias cq="condor_q"

alias vr='vi `find . -name requirements`'

alias-chkdep(){
	perl -n -e '@a=split / /, $_; printf "$_\n" for(@a) ;' $1 
}

