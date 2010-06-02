# === func-gen- : AbtViz/abtviz aberdeen fgp AbtViz/abtviz.bash fgn abtviz fgh AbtViz
abtviz-src(){      echo AbtViz/abtviz.bash ; }
abtviz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(abtviz-src)} ; }
abtviz-vi(){       vi $(abtviz-source) ; }
abtviz-env(){      echo -n  ; }
abtviz-usage(){
  cat << EOU
     abtviz-src : $(abtviz-src)
     abtviz-dir : $(abtviz-dir)

     abtviz-main 

        invoke the AbtViz main .. ideally should work 
           * from any dir 
           * with any DYLD_LIBRARY_PATH / LD_LIBRARY_PATH

         The main itself is a one line wrapper binary that execvp's 
         the python in the path ... done like this to allow attempts to
         bake in RPATH into the binary 

         issues with resources....
            Aberdeen_World_extract.root 


EOU
}
abtviz-dir(){ echo $(env-home)/AbtViz ; }
abtviz-cd(){  cd $(abtviz-dir); }
abtviz-mate(){ 
   cd
   mate $(abtbiz-dir) e/rootmq 
}


abtviz-main(){
  type $FUNCNAME
  local msg="=== $FUNCNAME :"
  local iwd=$PWD

  # move elsewhere to avoid python module shading issues 
  local tmp=/tmp/$USER/env/$FUNCNAME && mkdir -p $tmp 

  cd $tmp
  #cd $(env-home)/AbtViz

  local cmd="$(env-runenv) $(env-objdir)/abtviz/abtviz $(env-home)/AbtViz/ev.py"
  echo $msg $cmd 
  eval $cmd 

  cd $iwd
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


