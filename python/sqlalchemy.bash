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



    DOING SQLALCHEMY QUERIES IN A DJANGO MANNER


  * google:"SQLAlchemy session in  Django"


On SA sessions 
  * http://www.ibm.com/developerworks/aix/library/au-sqlalchemy/

Ambitious project that will never get there .. but undoubtably informative
  * http://code.google.com/p/django-sqlalchemy/wiki/Roadmap
  * http://bitbucket.org/lukaszb/django-alchemy/

Making SQLAlchemy models more compatible to Django (eg pagination ) 
  * http://reliablybroken.com/b/tag/sqlalchemy/

  * http://stackoverflow.com/questions/2546207/does-sqlalchemy-have-an-equivalent-of-djangos-get-or-create
{{{

That's basically the way to do it, there is no shortcut readily available AFAIK.

You could generalize it ofcourse:

def get_or_create(session, model, defaults=None, **kwargs):
    instance = session.Query(model.filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        params = dict((k, v) for k, v in kwargs.iteritems() if not isinstance(v, ClauseElement))
        params.update(defaults)
        instance = model(**params)
        session.add(instance)
        return instance, True

link|flag
	
edited Apr 6 at 22:27

	
answered Apr 6 at 17:47
WoLpH
6,999315
}}}

 * http://www.python-blog.com/tag/sqlalchemy/
    * allow do use sqlalchemy queries in a more djangoish manner (without having to pass the session around )
 
	You now can of course get comments using comment_set attribute on blog entry. But sometimes we want start querying for other objects, for some reasons.
	With Django ORM, no problem, just define method on the model and start querying ? it just works. With SQLAlchemy on the other hand we need to create query
	using Session object. But how does mapper use proper session if you call ?comment_set?? Well, I?ve found the answer here: http://www.sqlalchemy.org/docs/05/reference/orm/sessions.html . There is ?Session.object_session? class method which returns session object binded to an model object.

	So now we can simply define method for BlogComment
{{{
def get_related_comments(self):
    session = Session.object_session(self)
    comment_list = session.query(BlogComment)\
        .filter(BlogComment.entry_id==BlogEntry.id)\
        .all()
    return comment_list

}}}


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

