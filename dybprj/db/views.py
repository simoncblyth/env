from django.shortcuts import render_to_response
from django.template import RequestContext
from django.core.urlresolvers import reverse
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

from db.models import Database, Table, Column  # django model-of-model (mom) which are MySQLdb bootstrapped at syncdb time
#from env.sa import Session, DBISOUP            # SQLAlchemy models pulled from the soup 

from figs import make_fig
mimetype = dict( png="image/png", svg="image/svg+xml", pdf="application/pdf" )

#@login_required
def db_list(request):
    return render_to_response( "db/database_list.html" , { 'dblist':Database.objects.all() } , context_instance=RequestContext(request) )

#@login_required
def db_database(request, dbname ):
    db = Database.objects.get( name = dbname )
    tables = db.table_set.all()
    return render_to_response( "db/database_detail.html" , { 'db':db, 'tables':tables } , context_instance=RequestContext(request) )

#@login_required
def db_table(request, dbname , tabname ):
    """ next specifies to redirect back to table detail after posting comments """
    next = reverse( 'db-table' , kwargs={ 'dbname':dbname, 'tabname':tabname } ) 
    table = Table.objects.get( name = tabname, db__name = dbname )
    assert table , ( tabname, dbname )
    columns = table.column_set.all()
    return render_to_response( "db/table_detail.html" ,  { 'table':table , 'next':next, 'columns':columns } , context_instance=RequestContext(request) )

#@login_required
def db_column(request, dbname , tabname , colname ):
    next = reverse( 'db-column' , kwargs={ 'dbname':dbname, 'tabname':tabname, 'colname':colname } ) 
    table = Table.objects.get( name = tabname, db__name = dbname )
    columns = table.column_set.all()
    column = Column.objects.get( name = colname, table = table )
    return render_to_response( "db/column_detail.html" ,  { 'table':table, 'columns':columns, 'column':column, 'next':next, } , context_instance=RequestContext(request) )

#@login_required
def db_figure(request, **kwargs ):
    format = kwargs.pop('format','svg')
    response = HttpResponse(mimetype=mimetype[format])
    fig = make_fig( **kwargs )
    fig.savefig( response, format=format )
    return response









