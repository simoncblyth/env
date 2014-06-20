# === func-gen- : web/piwik fgp web/piwik.bash fgn piwik fgh web
piwik-src(){      echo web/piwik.bash ; }
piwik-source(){   echo ${BASH_SOURCE:-$(env-home)/$(piwik-src)} ; }
piwik-vi(){       vi $(piwik-source) ; }
piwik-env(){      elocal- ; }
piwik-usage(){ cat << EOU

PIWIK
======

Hosted web analytics

* http://piwik.org
* http://piwik.org/docs/requirements/#required-configuration-to-run-piwik

Needs:
 
#. MySQL (4.1+), 
#. PHP 5.3.2 (5.5 recommended) 
#. PHP extensions pdo, pdo_mysql, GD


EOU
}
piwik-dir(){ echo $(local-base)/env/web/web-piwik ; }
piwik-cd(){  cd $(piwik-dir); }
piwik-mate(){ mate $(piwik-dir) ; }
piwik-get(){
   local dir=$(dirname $(piwik-dir)) &&  mkdir -p $dir && cd $dir

}
