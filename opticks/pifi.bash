pifi-source(){   echo ${BASH_SOURCE} ; }
pifi-vi(){       vi $(pifi-source) ; }
pifi-env(){      elocal- ; }
pifi-usage(){ cat << EOU

PIFI plans, reports
=====================

Number of English words to Chinese characters
-----------------------------------------------

* https://www.actranslation.com/chinese/chinese-wordcount.htm

::

    1000 Chinese characters ~ 600-700 English words
    1000 English words ~ 1500-1700 Chinese characters

    800 Chinese characters ~ 480-560 English words  => aim for 500 words




EOU
}
pifi-dir(){ echo $(dirname $(pifi-source)) ;  }
pifi-cd(){  cd $(pifi-dir); }
pifi-c(){   cd $(pifi-dir); }
pifi-wc(){ pifi-cd ; wc -w *.rst ; }
pifi-edit(){ vi $(pifi-dir)/pifi_progress_report_aug2018.rst ; }


