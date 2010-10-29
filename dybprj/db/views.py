from django.shortcuts import render_to_response
from db.models import Database, Table
from django.template import RequestContext
from django.core.urlresolvers import reverse

ctx = {} 

def db_list(request):
    return render_to_response( "db/database_list.html" , { 'dblist':Database.objects.all() } , context_instance=RequestContext(request) )

def db_detail(request, dbname ):
    db = Database.objects.get( name = dbname )
    tables = db.table_set.all()
    return render_to_response( "db/database_detail.html" , { 'db':db, 'tables':tables } , context_instance=RequestContext(request) )

def db_table(request, dbname , tabname ):
    """ next specifies to redirect back to table detail after posting comments """
    table = Table.objects.get( name = tabname, db__name = dbname )
    next = reverse( 'db-table' , kwargs={ 'dbname':dbname, 'tabname':tabname } ) 
    return render_to_response( "db/table_detail.html" ,  { 'table':table , 'next':next } , context_instance=RequestContext(request) )

