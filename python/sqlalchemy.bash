# === func-gen- : python/sqlalchemy.bash fgp python/sqlalchemy.bash fgn sqlalchemy
sqlalchemy-src(){      echo python/sqlalchemy.bash ; }
sqlalchemy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(sqlalchemy-src)} ; }
sqlalchemy-vi(){       vi $(sqlalchemy-source) ; }
sqlalchemy-env(){      elocal- ; }
sqlalchemy-usage(){
  cat << EOU
     sqlalchemy-src : $(sqlalchemy-src)

     http://www.sqlalchemy.org

 
     sqlalchemu-version 

     sqlalchemy-dbcheck 
         verify the connection to DATABASE_URL by attempting 
         to dump the tables therein
 
     sqlalchemy 

    ||   || 0.4    ||  last version with Python 2.3 support. || 
    ||   || 0.5    ||  Python 2.4 or higher is required.     ||
    ||   || 0.6    ||                                        ||
    || C || 0.6.6  ||  hg tip                                ||         


EOU
}

sqlalchemy-srcfold(){ echo $(local-base)/env ; }
sqlalchemy-mode(){ echo 0.6 ; }
sqlalchemy-srcnam(){
   case ${1:-$(sqlalchemy-mode)} in
    0.5) echo sqlalchemy-0.5 ;;
    0.6) echo sqlalchemy ;;
   esac
}
sqlalchemy-srcdir(){  echo $(sqlalchemy-srcfold)/$(sqlalchemy-srcnam) ; }
#sqlalchemy-srcurl(){ echo http://svn.sqlalchemy.org/sqlalchemy/trunk@6065 ; }
#sqlalchemy-srcurl(){ echo http://svn.sqlalchemy.org/sqlalchemy/branches/rel_0_5 ; }

sqlalchemy-cd(){  cd $(sqlalchemy-srcdir) ;}

sqlalchemy-mate(){ mate $(sqlalchemy-srcdir) ; }

sqlalchemy-get(){
  local msg="=== $FUNCNAME :"
  local dir=$(sqlalchemy-srcfold)
  local nam=$(sqlalchemy-srcnam)
  mkdir -p $dir && cd $dir
  #[ ! -d "$nam" ] && svn co $(sqlalchemy-srcurl)  $nam || echo $msg $nam already exists in $dir skipping 

  hg clone http://hg.sqlalchemy.org/sqlalchemy 

}

sqlalchemy-ln(){
  local msg="=== $FUNCNAME :"
  python-ln $(sqlalchemy-srcdir)/lib/sqlalchemy sqlalchemy
  python-ln $(env-home) env
}

sqlalchemy-version(){ python -c "import sqlalchemy as _ ; print _.__version__ " ; }

sqlalchemy-dbcheck(){ $FUNCNAME- | python ; }
sqlalchemy-dbcheck-(){ cat << EOC
from private import Private
p = Private()
from sqlalchemy import create_engine
dburl = p('DATABASE_URL')
print dburl
db = create_engine(dburl)
print db.table_names()
EOC
}

