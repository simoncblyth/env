

mysql-env(){

  ## moved from .bash_mysql
  export MYSQL_HOME=$LOCAL_BASE/mysql
  export PATH=$MYSQL_HOME/bin:$PATH
}



mysql-env-not-pursued(){

  ## not pursued , already installed by .dmg 

   export MYSQL_NAME=mysql-5.0.45
   export MYSQL_CONF=-osx10.4-powerpc
   
   if [  -d $LOCAL_BASE/mysql ]; then
      echo ==== mysql-env $MYSQL_NAME $MYSQL_CONF 
   else
      sudo mkdir $LOCAL_BASE/mysql
      sudo chown $USER $LOCAL_BASE/mysql
   fi

}

mysql-get-not-pursued(){  

  ## not pursued , already installed by .dmg 


  #
  # http://dev.mysql.com/doc
  #   

  mysql-env
  
  cd $LOCAL_BASE/mysql

  local nam=${MYSQL_NAME}${MYSQL_CONF}
  local tgz=$nam.tar.gz
  
  ##local url=http://dev.mysql.com/get/Downloads/MySQL-5.0/$tgz/from/http://mysql.ntu.edu.tw/  redirected to the below
  local url=http://mysql.ntu.edu.tw/Downloads/MySQL-5.0/$tgz
  
  local ref=http://dev.mysql.com
  local cmd="curl -o $tgz   $url"
  
  echo $cmd
  test -f $tgz || eval $cmd
  test -d $nam || tar zxvf $tgz

}

