dbi-get(){

 #
 #   https://wiki.bnl.gov/dayabay/index.php?title=Database
 # 

 if [ -d $LOCAL_BASE/offline ]; then
    echo ==== dbi-get 
 else
    sudo mkdir $LOCAL_BASE/offline
    sudo chown $USER $LOCAL_BASE/offline 
 fi

 local dir=$LOCAL_BASE/offline/dbi
 mkdir -p $dir 
 
 cd $dir 
 local cmd="svn co $DYBSVN/db/trunk/ "

 echo $cmd
 eval $cmd

}

dbi-update(){

  local dir=$LOCAL_BASE/offline/dbi
  if [ -d "$dir" ]; then
       echo ==== dbi-update
  else
       echo ==== dbi-update ERROR dir $dir does not exist 
       return 1 
  fi

  cd $dir/trunk
  svn up 
}


dbi-build(){

   local dir=$LOCAL_BASE/offline/dbi
   cd $dir/trunk/DatabaseInterface/DbiTest/cmt
   cmt br cmt config
   cmt br cmt make
   
}




dbi-mysql(){
  $LOCAL_BASE/mysql/bin/mysql -u root -p$NON_SECURE_PASS 
}

dbi-mysql-databases(){
   echo "show databases;" | dbi-mysql 
}


dbi-reader(){
  $LOCAL_BASE/mysql/bin/mysql --local-infile=1 --user=dyreader --password=dyb_db dyb_offline
}

dbi-writer(){
  $LOCAL_BASE/mysql/bin/mysql --local-infile=1 --user=dywriter --password=$DYW_PASS dyb_offline
}

    




dbi-mysql-grants(){

local domain=%.phys.ntu.edu.tw

cat << EOG | dbi-mysql

create database dyb_temp;
create database dyb_offline;

grant select on dyb_offline.* to dyreader@localhost identified by "dyb_db";
grant ALL    on dyb_temp.*    to dyreader@localhost;
grant select on dyb_offline.* to dyreader@"$domain" identified by "dyb_db";
grant ALL    on dyb_temp.*    to dyreader@"$domain";

grant ALL    on dyb_offline.* to dywriter@localhost identified by "$DYW_PASS";
grant ALL    on dyb_temp.*    to dywriter@localhost;
grant ALL    on dyb_offline.* to dywriter@"$domain" identified by "$DYW_PASS";
grant ALL    on dyb_temp.*    to dywriter@"$domain";

flush privileges;

EOG



}