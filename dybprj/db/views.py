
from django.shortcuts import render_to_response
from db.models import Database, Table

def db_list(request):
    return render_to_response( "db/database_list.html" , dict( dblist=Database.objects.all() ) )

def db_detail(request, dbname ):
    return render_to_response( "db/database_detail.html" , dict( db=Database.objects.get(name=dbname) ) )

