# === func-gen- : tools/fast fgp tools/fast.bash fgn fast fgh tools
fast-src(){      echo tools/fast.bash ; }
fast-source(){   echo ${BASH_SOURCE:-$(env-home)/$(fast-src)} ; }
fast-vi(){       vi $(fast-source) ; }
fast-env(){      elocal- ; }
fast-usage(){ cat << EOU

FAST
=====

FAST is a set of tools for collecting, managing, and analyzing data about code performance.

* https://cdcvs.fnal.gov/redmine/projects/fast
* https://cdcvs.fnal.gov/redmine/attachments/5218/fast-concept.pdf
* https://cdcvs.fnal.gov/redmine/attachments/5219/fast-manual.pdf



EOU
}
fast-dir(){ echo $(local-base)/env/tools/fast ; }
fast-cd(){  cd $(fast-dir); }
fast-mate(){ mate $(fast-dir) ; }
fast-get(){
   local dir=$(dirname $(fast-dir)) &&  mkdir -p $dir && cd $dir


   local url=https://cdcvs.fnal.gov/redmine/attachments/download/4734/fast.tar.gz
   local tgz=$(basename $url)

   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "fast" ] && mkdir fast && tar zxvf $tgz -C fast       # exploding tarball

}
