from django.shortcuts import render_to_response


from django.template import RequestContext
from django.core.urlresolvers import reverse

from django.contrib.auth.decorators import login_required


## django models that are MySQLdb bootstrapped at syncdb time
from db.models import Database, Table

## sqlalchemy models pulled from the soup 
from env.sa import Session, DBISOUP


@login_required
def db_list(request):
    return render_to_response( "db/database_list.html" , { 'dblist':Database.objects.all() } , context_instance=RequestContext(request) )

@login_required
def db_detail(request, dbname ):
    db = Database.objects.get( name = dbname )
    tables = db.table_set.all()
    return render_to_response( "db/database_detail.html" , { 'db':db, 'tables':tables } , context_instance=RequestContext(request) )

@login_required
def db_table(request, dbname , tabname ):
    """ next specifies to redirect back to table detail after posting comments """
    table = Table.objects.get( name = tabname, db__name = dbname )
    next = reverse( 'db-table' , kwargs={ 'dbname':dbname, 'tabname':tabname } ) 
    return render_to_response( "db/table_detail.html" ,  { 'table':table , 'next':next, } , context_instance=RequestContext(request) )

@login_required
def db_column(request, dbname , tabname , colname ):
    table = Table.objects.get( name = tabname, db__name = dbname )
    column = Column.objects.get( name = tabname, table = table )
    next = reverse( 'db-column' , kwargs={ 'dbname':dbname, 'tabname':tabname, 'colname':colname } ) 
    return render_to_response( "db/column_detail.html" ,  { 'column':column, 'table':table , 'next':next, } , context_instance=RequestContext(request) )




from django.http import HttpResponse
from figs import demo_fig, column_fig

mimetype = dict( png="image/png", svg="image/svg+xml", pdf="application/pdf" )

@login_required
def db_table_fig(request, dbname , tabname , format ):
    response = HttpResponse(mimetype=mimetype[format])
    fig = demo_fig()
    fig.savefig( response, format=format )
    return response

@login_required
def db_column_fig(request, dbname , tabname , colname , format ):
    response = HttpResponse(mimetype=mimetype[format])
    fig = column_fig(dbname,tabname, colname )
    fig.savefig( response, format=format )
    return response










