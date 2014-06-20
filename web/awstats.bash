# === func-gen- : web/awstats fgp web/awstats.bash fgn awstats fgh web
awstats-src(){      echo web/awstats.bash ; }
awstats-source(){   echo ${BASH_SOURCE:-$(env-home)/$(awstats-src)} ; }
awstats-vi(){       vi $(awstats-source) ; }
awstats-env(){      elocal- ; apache- ; }
awstats-usage(){ cat << EOU

AWSTATS
=========

This log analyzer works as a CGI or from command line and 
shows you all possible information your log contains, in few graphical web pages. 

* http://awstats.sourceforge.net
* http://en.wikipedia.org/wiki/AWStats

* Perl 5.007003 or higher required to run AWStats 6.9 


Alternatives
--------------

* http://serverfault.com/questions/139343/command-line-tools-to-analyze-apache-log-files



Usage
------

Config File
~~~~~~~~~~~~~

Annoyingly awstats.pl does not look for config in invoking directory
so have to place config files at a hardcoded path. Create 
and set permissions for the hardcoded directory with::

   sudo mkdir -p /usr/local/etc/awstats
   sudo chown -R blyth:staff /usr/local/etc/awstats 

Configuration
~~~~~~~~~~~~~~

* http://awstats.sourceforge.net/docs/awstats_config.html


Run on 2 years of access_log
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:cms02 blyth$ awstats-;awstats-update
    /usr/local/env/web/awstats/data/cms02
    writing confpath /usr/local/etc/awstats/awstats.cms02.conf
    Create/Update database for config "/usr/local/etc/awstats/awstats.cms02.conf" by AWStats version 7.3 (build 20140126)
    From data in log file "/usr/local/env/web/awstats/data/cms02/access_log"...
    Warning: HostAliases parameter is not defined, awstats choose "dayabay.phys.ntu.edu.tw localhost 127.0.0.1".
    Phase 1 : First bypass old records, searching new record...
    Searching new records from beginning of log file...
    Phase 2 : Now process new records (Flush history on disk after 20000 hosts)...
    Flush history file on disk (unique url reach flush limit of 5000)
    Flush history file on disk (unique url reach flush limit of 5000)
    ...
    Flush history file on disk (unique url reach flush limit of 5000)
    Flush history file on disk (unique url reach flush limit of 5000)
    Jumped lines in file: 0
    Parsed lines in file: 6392219
     Found 750 dropped records,
     Found 0 comments,
     Found 0 blank records,
     Found 1155 corrupted records,
     Found 0 old records,
     Found 6390314 new qualified records.
    delta:cms02 blyth$ 

Create Reports
~~~~~~~~~~~~~~~

* http://localhost/env/web/awstats/awstats.cms02.html#month




EOU
}

awstats-name(){ echo awstats-7.3 ; }
awstats-dir(){ echo $(local-base)/env/web/awstats/$(awstats-name) ; }
awstats-datadir(){ echo $(local-base)/env/web/awstats/data ; }
awstats-cd(){  cd $(awstats-dir); }
awstats-mate(){ mate $(awstats-dir) ; }
awstats-get(){
   local dir=$(dirname $(awstats-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://prdownloads.sourceforge.net/awstats/$(awstats-name).tar.gz
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxvf $tgz
}

awstats-docs(){ open file://$(awstats-dir)/docs/index.html ; }

awstats-pl(){      perl $(awstats-dir)/wwwroot/cgi-bin/awstats.pl $* ; }
awstats-update(){  

   local iwd=$PWD
   local dir=$(awstats-sitedir)
   mkdir -p $dir && cd $dir

   echo $msg $PWD

   awstats-conf
   awstats-pl -config=$(awstats-site) -update ; 

   #cd $iwd
}


awstats-reportdir(){ echo $(apache-htdocs)/env/web/awstats ; }
awstats-report(){
   local reportdir=$(awstats-reportdir)
   mkdir -p $reportdir   
   local html=$reportdir/awstats.$(awstats-site).html

   echo $msg write report to $html

   awstats-pl -config=$(awstats-site) -output -staticlinks $* > $html
}

awstats-year(){ echo 2014 ; }
awstats-report-month(){ awstats-report -month=$1 -year=$(awstats-year) ; }
awstats-report-year(){  awstats-report -month=all -year=$(awstats-year) ; }


awstats-report-specific-(){
   local specific=${1:-alldomains}
   shift
   local msg="=== $FUNCNAME :"
   local reportdir=$(awstats-reportdir)
   mkdir -p $reportdir   
   local html=$reportdir/awstats.$(awstats-site).$specific.html
   echo $msg write report to $html

   awstats-pl -config=$(awstats-site) -output=$specific -staticlinks $* > $html
}
awstats-report-specific(){

   local names="alldomains allhosts lasthosts unknownip alllogins lastlogins allrobots lastrobots urldetail urlentry urlexit browserdetail osdetail unknownbrowser unknownos refererse refererpages keyphrases keywords errors404"

   for name in $names ; do 
       $FUNCNAME- $name $*
   done
}


awstats-report-specific-year(){ awstats-report-specific -month=all -year=$(awstats-year) ; }


awstats-sitedir-cd(){ cd $(awstats-sitedir) ; }
awstats-site(){      echo ${AWSTATS_SITE:-cms02} ; }
awstats-sitedir(){   echo $(awstats-datadir)/$(awstats-site) ; }
awstats-logpath(){   echo $(awstats-sitedir)/access_log ; }

awstats-confdir(){   echo /usr/local/etc/awstats ; } 
awstats-confpath(){  echo $(awstats-confdir)/awstats.$(awstats-site).conf ; }

awstats-conf(){
  local confpath=$(awstats-confpath) 
  echo $msg writing confpath $confpath  
  $FUNCNAME- > $confpath  
}
awstats-conf-(){  cat  << EOC

# http://awstats.sourceforge.net/docs/awstats_config.html

LogFile="$(awstats-logpath)"
LogType=W 
LogFormat=1 

# Possible values: 
# 0 - No DNS Lookup 
# 1 - DNS Lookup is fully enabled 
# 2 - DNS Lookup is made only from static DNS cache file (if it exists) 
# Default: 2 
# 
DNSLookup=2

DirData=$(awstats-sitedir)
SiteDomain="dayabay.phys.ntu.edu.tw" 

LogFormat = "%host %other %logname %time1 %methodurl %code %bytesd"

EOC
}



