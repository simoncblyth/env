# === func-gen- : trac/migration/trac2bitbucket fgp trac/migration/trac2bitbucket.bash fgn trac2bitbucket fgh trac/migration
trac2bitbucket-src(){      echo trac/migration/trac2bitbucket.bash ; }
trac2bitbucket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(trac2bitbucket-src)} ; }
trac2bitbucket-sdir(){     echo $(dirname $(trac2bitbucket-source)) ; }
trac2bitbucket-scd(){     cd $(trac2bitbucket-sdir) ; }
trac2bitbucket-vi(){       vi $(trac2bitbucket-source) ; }
trac2bitbucket-env(){      elocal- ; }
trac2bitbucket-usage(){ cat << EOU

Trac2Bitbucket 
==================

.. warning:: Other .bash of same name

Bitbucket documents an issue json format, that looks like
a good format to use to hold issue info in flexible manner.

* https://confluence.atlassian.com/pages/viewpage.action?pageId=330796872


EOU
}
trac2bitbucket-dir(){ echo $(local-base)/env/trac/migration/trac2bitbucket ; }
trac2bitbucket-cd(){  cd $(trac2bitbucket-dir); }
trac2bitbucket-c(){  cd $(trac2bitbucket-dir); }
trac2bitbucket-get(){
   local dir=$(dirname $(trac2bitbucket-dir)) &&  mkdir -p $dir && cd $dir

   hg clone https://bitbucket.org/secdev/trac2bitbucket

}

trac2bitbucket-wiki(){

   # [ "$NODE_TAG" != "C2" ] && echo $msg needs to run on server && return  
   # hmm python too old on C2
   #python- source
   #python $(trac2bitbucket-dir)/wiki.py 

   python $(trac2bitbucket-dir)/wiki.py --tracdir /tmp/t/env --output-dir /tmp/t/envhg


    # see also ~/env/trac/migration/tracwikidump.py  .sh
}



trac2bitbucket-tickets-json(){
   local tracdir=$1
   local name=$(basename $tracdir)
   local base=$(dirname $tracdir)
   local zip=$base/${name}_issues.zip
   local json=$base/${name}.json
   echo $json
}

trac2bitbucket-tickets(){
   local tracdir=${1:-/tmp/t/env}
   [ ! -f "$tracdir/db/trac.db" ] && echo $msg dir $tracdir is not a tracdir && return

   local name=$(basename $tracdir)
   local base=$(dirname $tracdir)
   local zip=$base/${name}_issues.zip
   local json=$base/${name}.json

   echo $msg converting tickets from $tracdir into bitbucket format zip $zip

   local cmd="python $(trac2bitbucket-dir)/tickets.py --tracdir $tracdir --output $zip"
   echo $msg $cmd
   eval $cmd

   unzip -l $zip

   echo $msg extrating json $json
   unzip -p $zip db-1.0.json > $json

   local sdir=$(trac2bitbucket-sdir)
   python $sdir/issues_json.py   $json

}

trac2bitbucket-tickets-check(){
   local tracdir=$1
   local json=$(trac2bitbucket-tickets-json $tracdir) 
   echo tracdir $tracdir json $json

}


