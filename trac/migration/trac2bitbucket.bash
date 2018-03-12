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


Got this to work, **but did not pursue** as
dont care much about wiki/ticket history for workflow. 
Instead adopted more manual approach of parsing the trac wikitext
and writing custom rst.


For Posterity only
-------------------

Succees to bring history of all trac wiki edits into a mercurial repo::

    python $(trac2bitbucket-dir)/wiki.py --tracdir  /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow --output-dir /tmp/w --create-repository --authordefault blyth


Doing hg serve and browsing find many entries with uninformative messages, propabably web wiki edits : better to name the wikipage in the message.
  
    http://simon.phys.ntu.edu.tw:8000



trac2bitbucket-tickets  /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow


simon:~ blyth$ trac2bitbucket-tickets  /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow
converting tickets from /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow into bitbucket format zip /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow_issues.zip
python /usr/local/env/trac/migration/trac2bitbucket/tickets.py --tracdir /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow --output /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow_issues.zip
Archive:  /var/scm/backup/g4pb/tracs/workflow/2015/04/28/235302/workflow_issues.zip
  Length     Date   Time    Name
 --------    ----   ----    ----
   471043  04-30-15 17:56   db-1.0.json
     6824  02-16-09 16:25   attachments/16/TMBackup.log
    25156  05-31-09 18:06   attachments/24/UnexpectedError-43.png
    29862  06-25-09 13:57   attachments/28/ipodsync.png
    27673  01-31-10 15:14   attachments/37/ingest-ingest-corrected.log






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


