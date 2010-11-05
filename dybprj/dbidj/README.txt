
= dbidj : django ORM with DBI tables =

  * i tried long ago in dybsite ...
    * but gave up and used SQLAlchemy 
  * bv did it in Rafiman ... 
  * BUT : only useful for limited subset of tables ...
    * payload tables : CPK (SEQNO, ROW_COUNTER )

Revisit to see if there have been any developments that make it a
possibilty now 

== introspection ==

Need a patch for inspectdb to work on mysql 4.1 
  * http://code.djangoproject.com/attachment/ticket/14618/introspection.py.diff

{{{
[blyth@cms01 django]$ svn diff
Index: db/backends/mysql/introspection.py
===================================================================
--- db/backends/mysql/introspection.py  (revision 14215)
+++ db/backends/mysql/introspection.py  (working copy)
@@ -1,4 +1,5 @@
 from django.db.backends import BaseDatabaseIntrospection
+from django.db.utils import DatabaseError 
 from MySQLdb import ProgrammingError, OperationalError
 from MySQLdb.constants import FIELD_TYPE
 import re
@@ -63,7 +64,7 @@
                     AND referenced_table_name IS NOT NULL
                     AND referenced_column_name IS NOT NULL""", [table_name])
             constraints.extend(cursor.fetchall())
-        except (ProgrammingError, OperationalError):
+        except (ProgrammingError, OperationalError,DatabaseError):
             # Fall back to "SHOW CREATE TABLE", for previous MySQL versions.
             # Go through all constraints and save the equal matches.
             cursor.execute("SHOW CREATE TABLE %s" %
self.connection.ops.quote_name(table_name))
}}}

== CPK ==

django still cannot handle composite-PK  ... 
  * fix introspected models such that each has a single PK
  * set Meta option to make unmanaged 

{{{
 ./manage.py inspectdb --database=prior | perl -p -e "s/primary_key=True, db_column='(ROW_COUNTER|dataVersion|parentPosition|className|objectId|schemaVersion)'/db_column='\$1'/" - > o.py 
 cat o.py | perl -p -e 's,class Meta:,class Meta:\n        managed = False,' - 
}}}


== google:"django ORM readonly" ==

  * http://www.mail-archive.com/django-users@googlegroups.com/msg46348.html 
     * suggests not to use ORM in readonly way 

  * http://efreedom.com/Question/1-3571634/Django-Using-Sqlalchemy-Read-Database

  * cannot relate the payload to vld as seqno is primary key on payload table
  * http://lethain.com/entry/2008/jul/23/replacing-django-s-orm-with-sqlalchemy/


== vdbi/vdbi/rumalchemy/repository.py ==

  * would be good to factor out the sqlalchemy introspection 




