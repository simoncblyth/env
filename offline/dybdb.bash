dybdb-vi(){  vi $BASH_SOURCE ; }
dybdb-usage(){
   cat << EOU
     In your .bash_profile define the "precursor" function "dybdb-" that sources this with 
        dybdb-(){ . path/to/dybdb.bash && dybdb-env $* ; }

     Subsequently can access these functions with dybdb-...

     Switch environment to that defined in a named section 
     of your ini file $(dybdb-ini)
 
        dybdb- testdb
        dybdb- offline_db
        dybdb- huangxt

        dybdb-info
        dybdb-sh    run mysql shell 
     

     To switch to a different config use 
        dybdb- <section>

     where section corresponds to a named section in your ini file $(dybdb-ini)

EOU
}

dybdb-dir(){ echo $(dirname $BASH_SOURCE) ; }
dybdb-ini(){ echo $HOME/.dybdb.ini ;  }
dybdb-edit(){ vi $(dybdb-ini) ; }
dybdb-sect(){ echo testdb ; }
dybdb-env(){
   local sect=${1:-$(dybdb-sect)}
   . $(dybdb-dir)/cfg.bash
   CFGDBG= cfg.parser $(dybdb-ini)
   eval "cfg.section.$sect"

}


dybdb-check(){
  local msg="=== $FUNCNAME "
  [ -z "$host" ]   && echo $msg host is not defined && return 1
  [ -z "$user" ]   && echo $msg user is not defined && return 1
  [ -z "$passwd" ] && echo $msg passwd is not defined && return 1
  [ -z "$db" ]     && echo $msg db is not defined && return 1
}

dybdb-sh-(){
   dybdb-check
   mysql --host=$host --user=$user --password=$passwd $*
}

dybdb-sh(){
   dybdb-check
   dybdb-sh- $db
}

dybdb-info(){
   cat << EOI

     $FUNCNAME   

        dybdb-ini : $(dybdb-ini)

          host : $host  
          user : $user  
        passwd : $passwd  
            db : $db  
  
      ENV_TSQL_URL  : $ENV_TSQL_URL
      ENV_TSQL_USER : $ENV_TSQL_USER
      ENV_TSQL_PSWD : $ENV_TSQL_PSWD

   (CAUTION THE ENV_TSQL_... ARE NOT EXPORTED BY DEFAULT )

     Exported :

EOI
   env | grep ENV_TSQL_
}

