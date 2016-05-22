# === func-gen- : tools/lldb/lldb fgp tools/lldb/lldb.bash fgn lldb fgh tools/lldb
lldb-src(){      echo tools/lldb/lldb.bash ; }
lldb-source(){   echo ${BASH_SOURCE:-$(env-home)/$(lldb-src)} ; }
lldb-vi(){       vi $(lldb-source) ; }
lldb-env(){      elocal- ; }
lldb-usage(){ cat << EOU





EOU
}
lldb-dir(){ echo $(local-base)/env/tools/lldb/tools/lldb-lldb ; }
lldb-cd(){  cd $(lldb-dir); }
lldb-mate(){ mate $(lldb-dir) ; }
lldb-get(){
   local dir=$(dirname $(lldb-dir)) &&  mkdir -p $dir && cd $dir

}
