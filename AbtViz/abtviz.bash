# === func-gen- : AbtViz/abtviz aberdeen fgp AbtViz/abtviz.bash fgn abtviz fgh AbtViz
abtviz-src(){      echo AbtViz/abtviz.bash ; }
abtviz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(abtviz-src)} ; }
abtviz-vi(){       vi $(abtviz-source) ; }
abtviz-env(){      echo -n  ; }
abtviz-usage(){
  cat << EOU
     abtviz-src : $(abtviz-src)
     abtviz-dir : $(abtviz-dir)


EOU
}
abtviz-dir(){ echo $(env-home)/AbtViz ; }
abtviz-cd(){  cd $(abtviz-dir); }
abtviz-mate(){ 
   cd
   mate $(abtbiz-dir) e/rootmq 
}



abtviz-ipython(){
   local msg="=== $FUNCNAME :"
   abtviz-cd
   env-
   local cmd="$(env-runenv) ipython $*"
   echo $msg $cmd ... will take a while if need to startup X11 
   eval $cmd 
}

abtviz-python(){
   local msg="=== $FUNCNAME :"
   abtviz-cd
   env-
   local cmd="$(env-runenv) python ev.py $*"
   echo $msg $cmd ... will take a while if need to startup X11 
   eval $cmd 
}

abtviz-gpython(){
   local msg="=== $FUNCNAME :"
   abtviz-cd
   env-
   local cmd="$(env-runenv) gdb $(which python)"
   echo $msg $cmd ... enter "set args ev.py" ... then run
   eval $cmd 
}
alias abtviz="abtviz-python"


