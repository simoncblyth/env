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


EOU
}

erlang-base(){      echo ${LOCAL_BASE:-$(local-base)}/env/erlang ; }
erlang-release(){   echo R12B-5 ; }
erlang-distname(){  echo otp_${1:-src}_$(erlang-release) ; }   
erlang-dir(){       echo $(erlang-base)/$(erlang-release)/src/$(erlang-distname) ; }
erlang-url(){       echo http://www.erlang.org/download/$(erlang-distname $1).tar.gz ; } 

erlang-cd(){        cd $(erlang-dir); }
erlang-mate(){      mate $(erlang-dir) ; }

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


