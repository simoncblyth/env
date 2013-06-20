# === func-gen- : scm/bitbucket/trac2bitbucket fgp scm/bitbucket/trac2bitbucket.bash fgn trac2bitbucket fgh scm/bitbucket
trac2bitbucket-src(){      echo scm/bitbucket/trac2bitbucket.bash ; }
trac2bitbucket-source(){   echo ${BASH_SOURCE:-$(env-home)/$(trac2bitbucket-src)} ; }
trac2bitbucket-vi(){       vi $(trac2bitbucket-source) ; }
trac2bitbucket-env(){      elocal- ; }
trac2bitbucket-usage(){ cat << EOU

Trac Tickets to Bitbucket zip
===============================

* :google:`bitbucket trac db-1.0.json`
* https://bitbucket.org/unayok/trac2bb
* https://confluence.atlassian.com/display/BITBUCKET/Export+or+Import+Issue+Data
* https://confluence.atlassian.com/pages/viewpage.action?pageId=330796872


Bitbucket issue ZIP import
-----------------------------

One time migration, clobbering any preexisting issues.


trac2bb
--------

* https://bitbucket.org/unayok/trac2bb

#. GPLv3 converter from trac.db into db-1.0.json 
#. does not handle attachements OR zip file creation
#. not very modular style, nevertheless interesting for the mappings 
   to translate trac table columns into JSON fields

trac2bitbucket
--------------

* https://bitbucket.org/thesheep/trac2bitbucket

#. operates via API https://api.bitbucket.org/1.0 rather than json file



EOU
}
trac2bitbucket-dir(){ echo $(local-base)/env/scm/bitbucket/trac2bb ; }
trac2bitbucket-cd(){  cd $(trac2bitbucket-dir); }
trac2bitbucket-mate(){ mate $(trac2bitbucket-dir) ; }
trac2bitbucket-get(){
   local dir=$(dirname $(trac2bitbucket-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://bitbucket.org/unayok/trac2bb.git 
}
