
pymysql-(){      . $ENV_HOME/db/pymysql.bash && pymysql-env $* ; }
pysqlite-(){     . $ENV_HOME/db/pysqlite.bash  && pysqlite-env  $* ; }
mysql-(){        . $ENV_HOME/db/mysql-.bash && mysql-env $* ; }     

