# === func-gen- : cpg/cpg fgp cpg/cpg.bash fgn cpg fgh cpg
cpg-src(){      echo cpg/cpg.bash ; }
cpg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cpg-src)} ; }
cpg-vi(){       vi $(cpg-source) ; }
cpg-env(){      elocal- ; }
cpg-usage(){
  cat << EOU
     cpg-src : $(cpg-src)
     cpg-dir : $(cpg-dir)

    Pre-requisites : 

     1) php present and enabled in apache and with mysql support enabled and gd library 
 
        Obtain with :
 
              sudo yum install php          ## this include apache conf in /etc/httpd/conf.d/php.conf   
              sudo yum install php-mysql

              sudo yum install gd
              sudo yum install php-gd

        Before installing  php-gd get warning :

             Your installation of PHP does not seem to include the 'GD' 
             graphic library extension and you have not indicated that you want to use ImageMagick. 
             Coppermine has been configured to use GD2 because the automatic GD detection sometimes fail. 
             If GD is installed on your system, the script should work else you will need to install ImageMagick.

        After installation AND an apache bounce :

             Your server supports the following image package(s): 
             GD Library version 1.x (gd1), GD Library version 2.x (gd2), the installer selected 'gd2'.


     2) mysql present and running 

           sudo /sbin/service mysqld start
           Starting MySQL:                    [  OK  ]

     3) Create database for cpg ...

           cpg-create-db
           cpg-mysql
              Welcome to the MySQL monitor.  Commands end with ; or \g.
              Your MySQL connection id is 4 to server version: 4.1.22
              Type 'help;' or '\h' for help. Type '\c' to clear the buffer.

           mysql> show tables ;
           Empty set (0.01 sec)

           mysql> quit

EOU
}
cpg-name(){ echo cpg14x ; }
cpg-dir(){ echo $(local-base)/env/cpg/$(cpg-name) ; }
cpg-cd(){  cd $(cpg-dir); }
cpg-get(){
   local dir=$(dirname $(cpg-dir)) &&  mkdir -p $dir && cd $dir
   local tgz=$(cpg-name).tar.gz
   [ ! -f "$tgz" ] && curl -L -O "http://downloads.sourceforge.net/project/coppermine/Coppermine/1.4.25%20%28stable%29/$tgz" 
   [ ! -d "$(cpg-name)" ] && tar zxvf $tgz


}

cpg-server(){ echo http://cms01.phys.ntu.edu.tw ; }
cpg-install(){

   cpg-get

   apache-
   apache-ln $(cpg-dir) cpg 
   apache-chown $(cpg-dir) -R

   sudo chcon -R -u system_u -t httpd_sys_content_t $(cpg-dir)

   ## open $(cpg-server)/cpg/install.php



}

cpg-val(){ echo $(private- ; private-val $*) ; }
cpg-mysql-(){    mysql --user $(cpg-val CPG_DATABASE_USER) --password=$(cpg-val CPG_DATABASE_PASSWORD) $1 ; }
cpg-create-db(){ echo "create database if not exists $(cpg-val CPG_DATABASE_NAME) ;"  | cpg-mysql- ; }
cpg-drop-db(){ echo   "drop   database if     exists $(cpg-val CPG_DATABASE_NAME) ;"  | cpg-mysql- ; }
cpg-mysql(){     cpg-mysql- $(cpg-val CPG_DATABASE_NAME) ; }


cpg-wipe(){
   local msg="=== $FUNCNAME :"
   local cmd="sudo rm -rf $(cpg-dir) "
   
   local ans
   read -p "$msg $cmd and drop the db : enter YES to proceed " ans
   [ "$ans" != "YES" ] && echo $msg skipping && return 0

   eval $cmd 
   cpg-drop-db

}

cpg-fromscratch(){

   cpg-wipe
   cpg-create-db
   cpg-install

}
