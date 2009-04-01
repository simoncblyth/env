invenio-src(){    echo invenio/invenio.bash ; }
invenio-source(){ echo ${BASH_SOURCE:-$(env-home)/$(invenio-src)} ; }
invenio-vi(){     vi $(invenio-source) ; }
invenio-env(){
  elocal-
}

invenio-usage(){

   cat << EOU 

     $(env-wikiurl)/Invenio
     $(env-wikiurl)/SELinux

     http://cdsware.cern.ch/invenio/index.html
     http://cdsware.cern.ch/download/INSTALL

      invenio-name : $(invenio-name)
      invenio-url  : $(invenio-url)


      invenio-createdb 
           create the mysql db tables needed for invenio


      invenio-apache-vhost-path : $(invenio-apache-vhost-path)
      invenio-apache-vhost
            generate the above file
      invenio-apache-vhost-
            cat to stdout 
            CAUTION this is a customisation of the apache conf written by
            `invenio-cfgbin` --apache-conf
            in order to debug SELinux and other issues            

      invenio-seprep
          emit proposed commands to setup the selinux labelling 
          ... if happy with them  pipe to sh
      
          RHEL guide to SELinux 
          (cms02) lynx /usr/share/doc/rhel-selg-en-4/index.html

      invenio-local--
          generate and propagate the local conf


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

invenio-installdir(){ echo $(invenio-dir)/install  ; }
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
invenio-servername(){  echo cms02.phys.ntu.edu.tw ; }
invenio-alias(){       echo cms02 ; }
invenio-server(){      echo http://$(invenio-servername) ; }


invenio-local--(){ sudo bash -lc "invenio-  ; invenio-local " ; }
invenio-local(){
   local msg="=== $FUNCNAME :"
   local path=`invenio-localconfpath`
   echo $msg generating $path 
   invenio-local- > $path 
   echo $msg propagating config change ... 
   `invenio-cfgbin` --update-all
}
invenio-local-(){
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

## CFG_APACHE_PASSWORD_FILE -- the file where Apache user credentials
## are stored.  Must be an absolute pathname.  If the value does not
## start by a slash, it is considered to be the filename of a file
## located under prefix/var/tmp directory.  This is useful for the
## demo site testing purposes.  For the production site, if you plan
## to restrict access to some collections based on the Apache user
## authentication mechanism, you should put here an absolute path to
## your Apache password file.
# CFG_APACHE_PASSWORD_FILE = /etc/httpd/conf/demo-site-apache-user-passwords

## CFG_APACHE_GROUP_FILE -- the file where Apache user groups are
## defined.  See the documentation of the preceding config variable.
# CFG_APACHE_GROUP_FILE = /etc/httpd/conf/demo-site-apache-user-groups


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

invenio-apacheconf-deprecated(){

   # must cp for the file to inherit the appropriate SELinux label of the destination directory 
   #
   sudo cp  $(invenio-installdir)/etc/apache/invenio-apache-vhost.conf  $(invenio-serverroot)/conf/
   sudo cp  $(invenio-installdir)/etc/apache/invenio-apache-vhost-ssl.conf  $(invenio-serverroot)/conf/
   sudo cp -R $(invenio-installdir)/var/www/* /var/www/html/ 
   ls -alZ $(invenio-serverroot)/conf/

}

invenio-errname(){ echo invenio_err.log ; }
invenio-logname(){ echo invenio.log ; }
invenio-abspath(){    
  case ${1:0:1} in
    "/") echo $1 ;;
      *) echo $(invenio-serverroot)/$1 ;;
  esac
}

invenio-logpath(){  invenio-abspath $(invenio-logdir)/$(invenio-logname) ; }
invenio-errpath(){  invenio-abspath $(invenio-logdir)/$(invenio-errname) ; }
invenio-ltail(){    sudo tail -f $(invenio-logpath) ; }
invenio-etail(){    sudo tail -f $(invenio-errpath) ; }

invenio-sitepkgs(){   echo /usr/lib/python2.3/site-packages ; }
invenio-serverroot(){ echo /etc/httpd ; }
invenio-docroot(){    echo $(invenio-installdir)/var/www ; }
invenio-logdir(){     echo logs ; }

#invenio-docroot(){  echo /var/www/html ; }
#invenio-logdir(){   echo $(invenio-installdir)/var/log ; }

invenio-apache-vhost-path(){ echo $(invenio-serverroot)/conf/invenio.conf ;  }
invenio-apache-vhost(){      sudo bash -lc "invenio- ; invenio-apache-vhost- > $(invenio-apache-vhost-path) " ; }
invenio-apache-vhost-(){

   cat << EOC
#
#  generated by $(invenio-source)::$FUNCNAME $(date)
#  which was invoked by invenio-apache-vhost
#  which created this file $(invenio-apache-vhost-path)
#

AddDefaultCharset UTF-8
ServerSignature Off
ServerTokens Prod
NameVirtualHost *:80
#Listen 80
<Files *.pyc>
   deny from all
</Files>
<Files *~>
   deny from all
</Files>
<VirtualHost *:80>
        ServerName $(invenio-servername)
        ServerAlias $(invenio-alias)
        ServerAdmin $(invenio-email)
        DocumentRoot $(invenio-docroot)
        <Directory $(invenio-docroot)>
           Options FollowSymLinks MultiViews
           AllowOverride None
           Order allow,deny
           allow from all
        </Directory>
        ErrorLog $(invenio-logdir)/$(invenio-errname)
        LogLevel warn
        CustomLog $(invenio-logdir)/$(invenio-logname) combined
        DirectoryIndex index.en.html index.html
        <LocationMatch "^(/+$|/index|/collection|/record|/author|/search|/browse|/youraccount|/youralerts|/yourbaskets|/yourmessages|/yourgroups|/submit|/getfile|/comments|/error|/oai2d|/rss|/help|/journal|/openurl|/stats)">
           SetHandler python-program
           PythonHandler invenio.webinterface_layout
           PythonDebug On
        </LocationMatch>
        <Directory $(invenio-docroot)>
           AddHandler python-program .py
           PythonHandler mod_python.publisher
           PythonDebug On
        </Directory>
</VirtualHost>


EOC


}


invenio-tmp-wipe(){
  local cmd="rm -rf $(invenio-tmp-files) "
  echo $cmd  
}

invenio-tmp-files(){

   cat << EOT
demobibdata.xml
demo-site-apache-user-groups
demo-site-apache-user-passwords
elmsubmit_tests_1.mbox
elmsubmit_tests_2.mbox
rec_fmt_20090331_141726.xml
TEST1.bfo
Test1.bft
TEST2.bfo
Test_2.bft
TEST3.bfo
Test3.bft
Test_no_template.test
tests_bibformat_elements
EOT

}


invenio-seprep(){

   local d=`invenio-installdir`
   invenio-selabel `invenio-installdir`
   cat << EOC
# the below should probably be tightened ?
sudo chcon -R -u system_u -t httpd_sys_content_t $d
sudo chcon -h -u system_u -t lib_t $(invenio-sitepkgs)/invenio   # the link
sudo chcon -R -u system_u -t httpd_sys_content_t $(invenio-installdir)/lib/python/invenio  # target of the link
EOC


}

invenio-selabel(){

  local d=$1
  local elem
  local dir
  echo $d | tr "/" "\n" | while read elem ; do
     if [ -n "$elem" ]; then
        dir="$dir/$elem"
        #echo $elem ... $dir
        #ls -lZ $dir
        local cmd="sudo chcon -u system_u -t var_t $dir "
        echo $cmd
     fi
  done
}




