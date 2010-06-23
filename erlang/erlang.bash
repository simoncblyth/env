# === func-gen- : erlang/erlang fgp erlang/erlang.bash fgn erlang fgh erlang
erlang-src(){      echo erlang/erlang.bash ; }
erlang-source(){   echo ${BASH_SOURCE:-$(env-home)/$(erlang-src)} ; }
erlang-vi(){       vi $(erlang-source) ; }
erlang-env(){      elocal- ; }
erlang-usage(){
  cat << EOU
     erlang-src : $(erlang-src)
     erlang-dir : $(erlang-dir)

    == erlang building ==

       http://www.erlang.org/download.html

    == Erlang EPEL versions ==

      ||  C  ||  EPEL4 ||   R11B-?   ||
      ||  N  ||  EPEL5 ||   R12B-?   ||  

            http://download.fedora.redhat.com/pub/epel/4/i386/repoview/erlang.html

   == erlang running ==

     google:"erlang command line"
        http://www.trapexit.org/Running_Erlang_Code_From_The_Command_Line

     erlang-- name
        compile(if necessary) and run a .erl module .. 
        entry point is the start function 

     erlang-vers

    == ERLANG TIPS ==

      1) man pages for modules
            erl -man io

      2) tab completion in the erl shell 



EOU
}

erlang-base(){      echo ${LOCAL_BASE:-$(local-base)}/env/erlang ; }
erlang-srcdir(){    echo ${ENV_HOME:-$(env-home)}/erlang ; }
erlang-beamdir(){   echo $(erlang-base)/ebin ; }
erlang-dir(){       echo $(erlang-base)/$(erlang-release)/src/$(erlang-distname) ; }
erlang-cd(){        cd $(erlang-dir); }
erlang-mate(){      mate $(erlang-dir) ; }

erlang-release(){   echo R12B-5 ; }
erlang-distname(){  echo otp_${1:-src}_$(erlang-release) ; }   
erlang-url(){       echo http://www.erlang.org/download/$(erlang-distname $1).tar.gz ; } 

erlang-scr(){       screen bash -c ". $(erlang-source) ; erlang-${1:-get} " ; }
erlang-get(){
   local dir=$(dirname $(dirname $(erlang-dir))) &&  mkdir -p $dir && cd $dir
   local t
   local dists="src doc_html doc_man" 
   for dist in $dists ; do 
      [ ! -f "$(erlang-distname $dist).tar.gz" ] && curl -O $(erlang-url $dist)
      [ ! -d "$dist"                           ] && mkdir $dist && tar zxvf $(erlang-distname $dist).tar.gz -C $dist
   done
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


