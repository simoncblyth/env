# === func-gen- : erlang/erl fgp erlang/erl.bash fgn erl fgh erlang
erl-src(){      echo erlang/erl.bash ; }
erl-source(){   echo ${BASH_SOURCE:-$(env-home)/$(erl-src)} ; }
erl-vi(){       vi $(erl-source) ; }
erl-env(){      elocal- ; }
erl-usage(){
  cat << EOU
     erl-src  : $(erl-src)
     erl-dir  : $(erl-dir)
     erl-ebin : $(erl-ebin)

     erl-erlpath name  :  $(erl-erlpath name) 
     erl-beampath name :  $(erl-beampath name) 

     erl-- name
       compile  erl-erlpath into erl-beampath and run  

       only compiles if necessary, entry point is the 
       start function in the module 
           http://www.trapexit.org/Running_Erlang_Code_From_The_Command_Line
     
     erl-vers
        example module using erl-- vers  
 
    == ERLANG TIPS ==

      1) man pages for modules
            erl -man io

    == ERL SHELL TIPS ==

      1) use tab completion 
      2) help().
      3) i().
  

EOU
}
erl-ebin(){ echo $(local-base)/env/ebin ; }
erl-dir(){ echo $(dirname $(erl-source)) ; }
erl-cd(){  cd $(erl-dir); }
erl-mate(){ mate $(erl-dir) ; }

erl-erlpath(){    echo $(erl-dir)/$1.erl ; }
erl-beampath(){   echo $(erl-ebin)/$1.beam ; }

erl--(){
   local msg="=== $FUNCNAME :"
   local name=${1:-vers}
   [ ! -d "$(erl-dir)" ] && mkdir -p $(erl-dir)
   local  erl=$(erl-erlpath $name)
   local beam=$(erl-beampath $name)
   [ ! -f "$erl"  ]  && echo $msg no .erl:$erl && return 1
   if [ ! -f "$beam" -o $erl -nt $beam ]; then 
       echo $msg compile $erl to  $beam 
       mkdir -p $(dirname $beam) && erlc -o $(dirname $beam) $erl 
   else
       echo $msg $beam is uptodate wrt $erl
   fi 
   erl -pa $(erl-ebin) -run $name -run init stop -noshell
}

erl-vers(){ erl-- vers ; }



