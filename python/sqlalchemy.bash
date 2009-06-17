# === func-gen- : python/sqlalchemy.bash fgp python/sqlalchemy.bash fgn sqlalchemy
sqlalchemy-src(){      echo python/sqlalchemy.bash ; }
sqlalchemy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sqlalchemy-src)} ; }
sqlalchemy-vi(){       vi $(sqlalchemy-source) ; }
sqlalchemy-env(){      elocal- ; }
sqlalchemy-usage(){
  cat << EOU
     sqlalchemy-src : $(sqlalchemy-src)

     http://www.sqlalchemy.org

     sqlalchemy-dbcheck 
         verify the connection to the database 


EOU
}

sqlalchemy-srcfold(){ echo $(local-base)/env ; }
sqlalchemy-mode(){ echo 0.5 ; }
sqlalchemy-srcnam(){
   case ${1:-$(sqlalchemy-mode)} in
    0.5) echo sqlalchemy-0.5 ;;
   esac
}
sqlalchemy-srcdir(){  echo $(sqlalchemy-srcfold)/$(sqlalchemy-srcnam) ; }
sqlalchemy-srcurl(){ echo http://svn.sqlalchemy.org/sqlalchemy/trunk@6065 ; }
sqlalchemy-get(){
  local msg="=== $FUNCNAME :"
  local dir=$(sqlalchemy-srcfold)
  local nam=$(sqlalchemy-srcnam)
  mkdir -p $dir && cd $dir
  [ ! -d "$nam" ] && svn co $(sqlalchemy-srcurl)  $nam || echo $msg $nam already exists in $dir skipping 
}

sqlalchemy-ln(){
  local msg="=== $FUNCNAME :"
  python-ln $(sqlalchemy-srcdir)/lib/sqlalchemy sqlalchemy
  python-ln $(env-home) env
}

sqlalchemy-version(){ python -c "import sqlalchemy as _ ; print _.__version__ " ; }

sqlalchemy-dbcheck(){ $FUNCNAME- | python ; }
sqlalchemy-dbcheck-(){ cat << EOC
from env.base.private import Private
p = Private()
from sqlalchemy import create_engine
db = create_engine( p('DATABASE_URL') )
print db.table_names()
EOC
}

