from django.conf import settings
from django.shortcuts import render_to_response
from db.models import Database, Table
from django.template import RequestContext
from django.core.urlresolvers import reverse

ctx = dict( MEDIA_URL=settings.MEDIA_URL )

def db_list(request):
    return render_to_response( "db/database_list.html" , dict( ctx, dblist=Database.objects.all() ) , context_instance=RequestContext(request) )

def db_detail(request, dbname ):
    db = Database.objects.get(name=dbname)
    tables = db.table_set.all()
    return render_to_response( "db/database_detail.html" , dict( ctx, db=db, tables=tables ) , context_instance=RequestContext(request) )

def db_table(request, dbname , tabname ):
    tab = Table.objects.get(name=tabname, db__name=dbname )
    next = reverse( 'db-table' , kwargs=dict(dbname=dbname, tabname=tabname) ) ## where to go after posting comment
    return render_to_response( "db/database_table_detail.html" , dict( ctx, tab=tab , next=next ) , context_instance=RequestContext(request) )

