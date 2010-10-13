"""
   Note this is not standalone django usage as need to hookup to Users in DB

   Idea is to pony off :
      * Django AuthUserFileBackend extrapolated Trac/SVN Users
      * Django Admin managed Groups 

   To generate MySQL GRANT/REVOKE sql, thereby allowing further
   extrapolation of Trac/SVN users to MySQL 



   NEXT STEPS ...
       * how to capture permission groups for DBI table pair  updating 
       * introspect current MySQL GRANTS ?

       * propagate Dbi table names in DBCONFed DB into Django Groups ?
       * apply generated sql to DB pointed to by DBCONF via MySQLdb ?


"""

import os
os.environ.update( DJANGO_SETTINGS_MODULE='runinfo.settings' )

from django.template import Context, Template
from django.contrib.auth.models import User

dbg = True
tdir = os.path.join( os.path.dirname( __file__ ) , "templates" )

if __name__=='__main__':

    extras = {}
    tmpl = os.path.join( tdir, "genuser.sql" )  
    t = Template( file(tmpl).read() )
    c = Context( dict( users=User.objects.exclude(password__startswith="sha")))     ## exclude native Django Users 
    c.update( extras )
    print t.render( c )



