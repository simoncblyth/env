
    ## using SQLAlchemy querying to determine the extents of the table fields ...
    ## and filling in inline SVG in the template ...
    ##   ... this approach does not scale to slow DB queries

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




def db_table_dev(request, dbname , tabname ):
    """ next specifies to redirect back to table detail after posting comments """
    table = Table.objects.get( name = tabname, db__name = dbname )
    next = reverse( 'db-table-dev' , kwargs={ 'dbname':dbname, 'tabname':tabname } ) 
    return render_to_response( "db/table_detail.html" ,  { 'table':table , 'next':next, 'count':count, 'objs':objs, 'vld':vld , 'viewBox':viewBox } , context_instance=RequestContext(request) )




## inline svg  ... low level approach ... shelved as too low a level 
"""
{% autoescape off %}
<svg  width="500px" height="500px"  viewBox="{{ viewBox }}"
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve">
   <g id="root" >

       {% for obj in objs %}
          {{ obj|svgline }}
       {% endfor %}

  </g>
</svg>
{% endautoescape %}

"""

## table duming 
"""
   <h1> viz {{ table.name }} count {{ count }}  </h1>
   {% if vld %}
      <table>
         <tr>
            <th> seqno </th>
            <th> start </th>
            <th> end </th>
            <th> version </th>
            <th> insert </th>
         </tr>
      {% for obj in objs %}
         <tr>
             <td> {{ obj.SEQNO }} </td>
             <td> {{ obj.TIMESTART }} </td>
             <td> {{ obj.TIMEEND }} </td>
             <td> {{ obj.VERSIONDATE }} </td>
             <td> {{ obj.INSERTDATE }} </td>
         </tr> 
      {% endfor %}
     </table>
   {% endif %} 
"""

