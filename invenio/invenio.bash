invenio-src(){    echo invenio/invenio.bash ; }
invenio-source(){ echo ${BASH_SOURCE:-$(env-home)/$(invenio-src)} ; }
invenio-vi(){     vi $(invenio-source) ; }
invenio-env(){
  elocal-
}

invenio-usage(){

   cat << EOU 

     $(env-wikiurl)/Invenio

     http://cdsware.cern.ch/invenio/index.html
     http://cdsware.cern.ch/download/INSTALL


      invenio-name : $(invenio-name)
      invenio-url  : $(invenio-url)


EOU

}

invenio-name(){     echo cds-invenio-0.99.1 ; }
invenio-basename(){ echo $(invenio-name).tar.gz ; }
invenio-url(){  echo http://cdsware.cern.ch/download/$(invenio-basename) ; }
invenio-dir(){  echo $(local-base)/env/invenio ; }
invenio-cd(){   cd $(invenio-dir)/$1 ; }

invenio-get(){
  mkdir -p $(invenio-dir)
  invenio-cd
  [ ! -f "$(invenio-basename)" ] && curl -O  $(invenio-url)
  [ ! -f "$(invenio-basename).md5" ] && curl -O  $(invenio-url).md5
  [ ! -f "$(invenio-basename).sig" ] && curl -O  $(invenio-url).sig
  [ ! -d "$(invenio-name)"     ] && tar zxvf "$(invenio-basename)" 

  md5sum --check $(invenio-basename).md5


}


invenio-installdir(){
  echo $(invenio-dir)/install 
}

invenio-configure(){

  invenio-cd $(invenio-name)
  local pfx=$(invenio-installdir) && mkdir -p $pfx
  ./configure --prefix=$pfx --with-python=/usr/bin/python --with-mysql=/usr/bin/mysql
  
}

invenio-confpath(){      echo  `invenio-installdir`/etc/invenio.conf ; } 
invenio-localconfpath(){ echo  `invenio-installdir`/etc/invenio-local.conf ;  }
invenio-conf(){          vi $(invenio-confpath) ; }
invenio-localconf(){     vi $(invenio-localconfpath) $(invenio-confpath) ; }
invenio-cfgbin(){       echo  `invenio-installdir`/bin/inveniocfg ; }
invenio-cfg(){         `invenio-cfgbin` $* ; }   

invenio-email(){       echo simon.c.blyth@gmail.com ; }
invenio-server(){      echo http://cms02.phys.ntu.edu.tw ; }

invenio-local(){
  cat << EOL

[Invenio]

## CFG_SITE_URL - specify URL under which your installation will be
## visible.  For example, use "http://your.site.com".  Do not leave
## trailing slash.
CFG_SITE_URL = $(invenio-server)
CFG_SITE_ADMIN_EMAIL = $(invenio-email)
CFG_SITE_SUPPORT_EMAIL = $(invenio-email)

CFG_DATABASE_HOST = localhost
CFG_DATABASE_NAME = cdsinvenio
CFG_DATABASE_USER = cdsinvenio
CFG_DATABASE_PASS = $(private- ; private-val INVENIO_PASS)


EOL

}

invenio-createdb(){
   mysql -h localhost -u root -p $(private- ; private-val MYSQL_ROOT_PASS)  <<  EOC
CREATE DATABASE cdsinvenio DEFAULT CHARACTER SET utf8;    
GRANT ALL PRIVILEGES ON cdsinvenio.*  TO cdsinvenio@localhost IDENTIFIED BY '$(private- ; private-val INVENIO_PASS)';          
EOC

}


invenio-apache(){
  
  local dir=`invenio-installdir`
  local apachegroup=apache
  sudo chown -R apache:apache $dir
  #sudo chmod -R g+r $dir
  #sudo chmod -R g+rw $dir/var
  sudo find $dir -type d -exec chmod g+rxw {} \;

}


