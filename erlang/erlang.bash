# === func-gen- : erlang/erlang fgp erlang/erlang.bash fgn erlang fgh erlang
erlang-src(){      echo erlang/erlang.bash ; }
erlang-source(){   echo ${BASH_SOURCE:-$(env-home)/$(erlang-src)} ; }
erlang-vi(){       vi $(erlang-source) ; }
erlang-env(){      elocal- ; }
erlang-usage(){
  cat << EOU
     erlang-src : $(erlang-src)
     erlang-dir : $(erlang-dir)

     google:"erlang command line"
        http://www.trapexit.org/Running_Erlang_Code_From_The_Command_Line

     erlang-- name
        compile(if necessary) and run a .erl module .. 
        entry point is the start function 

     erlang-vers



    == Erlang Versions ==


      ||  C  ||   R11B   ||
      ||  N  ||   R12B   ||  



    == ERLANG TIPS ==

      1) man pages for modules
            erl -man io

      2) tab completion in the erl shell 



EOU
}
erlang-srcdir(){  echo $(env-home)/erlang ; }
erlang-beamdir(){ echo $(local-base)/env/erlang ; }
erlang-dir(){ echo $(local-base)/env/erlang/erlang-erlang ; }
erlang-cd(){  cd $(erlang-dir); }
erlang-mate(){ mate $(erlang-dir) ; }
erlang-get(){
   local dir=$(dirname $(erlang-dir)) &&  mkdir -p $dir && cd $dir
}

erlang--(){
   local msg="=== $FUNCNAME :"
   local name=${1:-vers}
   [ ! -d "$(erlang-beamdir)" ] && mkdir -p $(erlang-beamdir)
   local  erl=$(erlang-srcdir)/$name.erl
   local beam=$(erlang-beamdir)/$name.beam
   [ ! -f "$erl"  ]  && echo $msg no .erl:$erl && return 1
   [ ! -f "$(erlang-beamdir)/$name.beam" -o $erl -nt $beam ] && erlc -o $(erlang-beamdir) $erl 
   erl -pa $(erlang-beamdir) -run $name -run init stop -noshell
}

erlang-vers(){ erlang-- vers ; }


