# === func-gen- : mysql/mysql-python fgp mysql/mysql-python.bash fgn mysql-python fgh mysql
mysql-python-src(){      echo mysql/mysql-python.bash ; }
mysql-python-source(){   echo ${BASH_SOURCE:-$(env-home)/$(mysql-python-src)} ; }
mysql-python-vi(){       vi $(mysql-python-source) ; }
mysql-python-env(){      elocal- ; }
mysql-python-usage(){
  cat << EOU
     mysql-python-src : $(mysql-python-src)
     mysql-python-dir : $(mysql-python-dir)




    mysql-python-*


          http://mysql-python.blogspot.com/
               

          normally can just 
              pip install mysql-python

              BUT CAUTION WRT WHICH mysql-config IS IN PATH ... 
              it dictates which mysql/python to build against



EOU
}


mysql-python-name(){ echo MySQLdb-2.0 ;}
mysql-python-dir(){ echo $(local-base)/env/mysql/$(mysql-python-name) ; }
mysql-python-cd(){  cd $(mysql-python-dir); }
mysql-python-mate(){ mate $(mysql-python-dir) ; }
mysql-python-get(){
   local dir=$(dirname $(mysql-python-dir)) &&  mkdir -p $dir && cd $dir
   hg clone http://mysql-python.hg.sourceforge.net/hgweb/mysql-python/$(mysql-python-name)/
}

mysql-python-which(){
   which python
   which mysql_config
}


mysql-python-install(){
   mysql-python-cd
   mysql-python-which

   python setup.py install

}




mysql-python-version(){    python -c "import MySQLdb as _ ; print _.__version__ " ; }
mysql-python-installdir(){ python -c "import os,MySQLdb as _ ; print os.path.dirname(_.__file__) " ; }
mysql-python-info(){ cat << EOI
    hostname   : $(hostname)
    version    : $(mysql-python-version)
    installdir : $(mysql-python-installdir)
EOI
}




#mysql-python-dir(){ echo $(local-base)/mysql-python/MySQL-python-$(mysql-python-ver) ; }
#mysql-python-ver(){ echo 1.2.3c1 ; }
#mysql-python-ver(){ echo 1.2.2 ; }
#mysql-python-tgz(){ echo MySQL-python-$(mysql-python-ver).tar.gz ; }
#mysql-python-url(){ echo http://downloads.sourceforge.net/project/mysql-python/mysql-python/$(mysql-python-ver)/$(mysql-python-tgz) ; }


#mysql-python-cd(){  cd $(mysql-python-dir) ; }
#mysql-python-get(){  
#
#   local dir=$(dirname  $(mysql-python-dir))
#   local nam=$(basename $(mysql-python-dir))
#   mkdir -p $dir && cd $dir   
# 
#   local tgz=$(mysql-python-tgz)
#   [ ! -f "$tgz" ] && curl -L -O $(mysql-python-url) 
#   [ ! -d "$nam" ] && tar zxvf $tgz
#}



