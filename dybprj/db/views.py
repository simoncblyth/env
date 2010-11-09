from django.shortcuts import render_to_response
from db.models import Database, Table
from django.template import RequestContext
from django.core.urlresolvers import reverse

from django.contrib.auth.decorators import login_required


## sqlalchemy 
from env.sa import Session, DBISOUP

import time
epoch = lambda dt:time.mktime(dt.timetuple())

"""
  http://matplotlib.sourceforge.net/ 
  http://code.creativecommons.org/svnroot/stats/reports/temp/date_demo.py


  http://matplotlib.sourceforge.net/users/artists.html#figure-container 
     line drawing example can be the basis of the viz i have in mind

     help(matplotlib.dates) help(matplotlib.ticker)


    sqlalchemy -- numpy -- matplotlib -- django


     https://github.com/dalloliogm/sqlalchemy-recarray
     http://www.sqlalchemy.org/trac/ticket/1572

"""

ctx = {} 

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

    ## SQLAlchemy ...
    session = Session()
    kls = DBISOUP.get(tabname, None)
    assert kls, "no SA class for tabname %s " % tabname
    assert hasattr( kls, 'SEQNO' ) , kls

    ## aggregates dont need ORM 
    from sqlalchemy import select, func
    tc = kls._table.c
    count,seqmin,seqmax,tmin,tmax = select([func.count(),func.min(tc.SEQNO),func.max(tc.SEQNO),func.min(tc.TIMESTART),func.max(tc.TIMEEND),]).where(tc.TIMESTART>datetime(2000, 1, 1, 0, 0, 0)).execute().fetchone()
    viewBox = "%s %s %s %s" % ( epoch(tmin), seqmin , epoch(tmax), seqmax )

    limit = 25
    pay = hasattr( kls, 'ROW_COUNTER' )
    vld = not pay  
    if pay: 
        objs = session.query(kls).order_by(kls.SEQNO,kls.ROW_COUNTER).all()[0:limit]
    else:
        objs = session.query(kls).order_by(kls.SEQNO).all()[0:limit]

   
    return render_to_response( "db/table_detail.html" ,  { 'table':table , 'next':next, 'count':count, 'objs':objs, 'vld':vld , 'viewBox':viewBox } , context_instance=RequestContext(request) )

