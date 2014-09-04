# === func-gen- : b2c/b2c fgp b2c/b2c.bash fgn b2c fgh b2c
b2c-src(){      echo b2c/b2c.bash ; }
b2c-source(){   echo ${BASH_SOURCE:-$(env-home)/$(b2c-src)} ; }
b2c-vi(){       vi $(b2c-source) ; }
b2c-env(){      elocal- ; }
b2c-usage(){ cat << EOU

b2c STATICS FROM BACKUP
=========================


FUNCTIONS
----------

b2c-create-tarball
       create tarball with a selection of the 13G that comprise /data/heprez on cms01.phys.ntu.edu.tw




::


    [blyth@cms01 conf]$ du -hs /data/heprez
    13G /data/heprez

    [blyth@cms01 ~]$ cd /data/heprez

    [blyth@cms01 heprez]$ du -hs * 
    9.9G    data
    2.6G    install
    475M    install.aug5-2008

    [blyth@cms01 data]$ du -hs * 
    1.6G    backup                         ## xmldb level backups
    520M    images
    7.9G    scrape

    [blyth@cms01 data]$ du -hs images/*    ##   
    50M     images/apache
    260K    images/applescript
    8.0K    images/du.txt
    6.0M    images/extras
    964K    images/fonts
    13M     images/indices
    176K    images/maybe-trash
    2.1M    images/pdfpng2apache
    119M    images/pdg
    300K    images/pdgparse
    50M     images/pdgs
    97M     images/qtags
    183M    images/svgsmry

    [blyth@cms01 data]$ du -hs scrape/*
    101M    scrape/20060122
    101M    scrape/20060122-nohtm
    50M     scrape/20060122-nohtm.tar.gz
    69M     scrape/20060723
    66M     scrape/20060724
    9.4M    scrape/20060803
    243M    scrape/20060915
    209M    scrape/20061001
    230M    scrape/20061005
    114M    scrape/20061005.tar.gz
    231M    scrape/20061109
    210M    scrape/20061109-relative
    110M    scrape/20061109.tar.gz
    44M     scrape/20070404
    227M    scrape/20070406
    110M    scrape/20070406.tar.gz
    245M    scrape/20080401
    247M    scrape/20080701
    266M    scrape/20090530
    271M    scrape/20100204
    272M    scrape/20100205
    267M    scrape/20100307
    264M    scrape/20100418
    264M    scrape/20100426
    289M    scrape/20111116
    304M    scrape/20111217
    300M    scrape/20120306
    311M    scrape/20120422
    307M    scrape/20120422.keep
    309M    scrape/20120510
    307M    scrape/20120514
    239M    scrape/20120524
    375M    scrape/20130325
    362M    scrape/20130617
    312M    scrape/20130711
    313M    scrape/20130716
    8.0K    scrape/du2.txt
    8.0K    scrape/du.txt
    117M    scrape/may2-20080401.tar.gz

    [blyth@cms01 heprez]$ l install/
    total 128
    drwxrwxr-x  5 blyth blyth 4096 Jun 19  2013 chibacvs
    drwxrwxr-x  6 blyth blyth 4096 Apr 13  2012 xsd2xhtml
    drwxrwxr-x  4 blyth blyth 4096 Dec  5  2011 apache
    drwxrwxr-x  6 blyth blyth 4096 Nov 21  2011 xsd2xhtml.nov2011
    drwxrwxr-x  6 blyth blyth 4096 Jan 11  2010 xsd2xhtml.jan2010
    drwxrwxr-x  4 blyth blyth 4096 May  6  2009 chibatomcat
    drwxrwxr-x  3 blyth blyth 4096 Apr 30  2009 antjython
    drwxrwxr-x  3 blyth blyth 4096 Apr 30  2009 jython
    drwxrwxr-x  4 blyth blyth 4096 Apr 30  2009 tomcat
    drwxrwxr-x  3 blyth blyth 4096 Apr 30  2009 cocoon
    drwxrwxr-x  4 blyth blyth 4096 Apr 30  2009 eperl
    drwxrwxr-x  3 blyth blyth 4096 Apr 30  2009 log4j
    drwxrwxr-x  3 blyth blyth 4096 Apr 30  2009 junit
    drwxrwxr-x  3 blyth blyth 4096 Apr 30  2009 jythonbuild
    drwxrwxr-x  3 blyth blyth 4096 Apr 29  2009 exist
    drwxrwxr-x  5 blyth blyth 4096 Apr 29  2009 ant



[blyth@cms01 heprez]$ du -hs heprez-20140904.tar.gz 
775M    heprez-20140904.tar.gz

[blyth@cms01 heprez]$ mv heprez-20140904.tar.gz /var/www/html/downloads/


change selinux context to make visible at below url
-----------------------------------------------------

* http://cms01.phys.ntu.edu.tw/downloads/heprez-20140904.tar.gz


::

    [blyth@cms01 downloads]$ ls -Z heprez-20140904.tar.gz 
    -rw-rw-r--  blyth    blyth    user_u:object_r:var_t            heprez-20140904.tar.gz
    [blyth@cms01 downloads]$ ls -Z b2charm_end_of_2011_v012.tar.gz 
    -rw-r--r--  blyth    blyth    user_u:object_r:httpd_sys_content_t b2charm_end_of_2011_v012.tar.gz
    [blyth@cms01 downloads]$ 
    [blyth@cms01 downloads]$ sudo chcon --reference b2charm_end_of_2011_v012.tar.gz heprez-20140904.tar.gz 
    Password:
    [blyth@cms01 downloads]$ ls -Z heprez-20140904.tar.gz 
    -rw-rw-r--  blyth    blyth    user_u:object_r:httpd_sys_content_t heprez-20140904.tar.gz




EOU
}
b2c-dir(){ echo $(local-base)/env/b2c/b2c-b2c ; }
b2c-cd(){  cd $(b2c-dir); }

b2c-stamp(){ echo $(date "+%Y%m%d") ; }
b2c-bkpd(){  echo /data/heprez ; }
b2c-create-tarball(){
   local msg="=== $FUNCNAME :"

   local tag=${1:-20130716}
   local bkpd=$(b2c-bkpd)

   [ "$(hostname)" != "cms01.phys.ntu.edu.tw" ] && echo $msg THIS IS FOR RUNNING ON cms01 && return
   [ "$(pwd)" != "$bkpd" ] && echo $msg RUN THIS FROM $bkpd && return 

   tar cvfz heprez-$(b2c-stamp).tar.gz  \
            data/scrape/$tag \
            data/images \
            data/backup \
            install

}


