# === func-gen- : offline/dcs fgp offline/dcs.bash fgn dcs fgh offline
dcs-src(){      echo offline/dcs.bash ; }
dcs-source(){   echo ${BASH_SOURCE:-$(env-home)/$(dcs-src)} ; }
dcs-vi(){       vi $(dcs-source) ; }
dcs-env(){      elocal- ; }
dcs-usage(){
  cat << EOU
     dcs-src : $(dcs-src)
     dcs-dir : $(dcs-dir)


     dcs-get
           create "dcs" database and populate with the no-data dump
           the dump was created with 
                 mysqlsdump -d > dcs.sql 
           after clearing "database" settings from ~/.my.cnf 
 


EOU
}
dcs-dir(){ echo $(local-base)/env/offline/offline-dcs ; }
dcs-cd(){  cd $(dcs-dir); }
dcs-mate(){ mate $(dcs-dir) ; }

dcs-dumppath(){ echo $HOME/dcs.sql ; }
dcs-get(){
   local dir=$(dirname $(dcs-dir)) &&  mkdir -p $dir && cd $dir

   echo "create database if not exists dcs " | mysql   
   cat $(dcs-dumppath) | mysql dcs

}




