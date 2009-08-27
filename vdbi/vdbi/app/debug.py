

class Dump(dict):
    """
         Dump the alternative implementations available for each of the apps component
         keys ... adding more implemtations in egg entry_points 
         and picking them via '''use''' allows major changes to app beahviour to be made without touching the rum code 
    """
    def __init__(self, app ):
        from rum.component import Component
        import pkg_resources as pr
        cls = app.__class__
        comps =  [getattr(cls,x) for x in dir(cls) if not(x.startswith('_')) and isinstance( getattr(cls, x) , Component)  ]
        for comp in comps:
            print "=========== %s ============== " % comp.key 
            print list(pr.iter_entry_points(comp.key))
            self[comp.key] = list(pr.iter_entry_points(comp.key))

class Mapr(dict):
    """
        For checking on the mappers that the app is using 
    """
    def __init__(self, app):
        from rumalchemy.util import get_mapper
        for cls in app.resources.keys():
            if hasattr(cls._table.c, 'SEQNO' ):
                print repr(cls._table.c.SEQNO)
            mapr = get_mapper(cls)
            self[cls.__name__] = mapr
            print mapr.c.keys()

class Qry(dict):
    """
        For doing test queries using the session etc... hooked up into the app 

      In [2]: q['Simpmtspec']
      Out[2]: <sqlalchemy.orm.query.Query object at 0x1e31f50>
      In [3]: q['Simpmtspec'].first()
      In [4]: q['Simpmtspec'].count()
      Out[4]: 3169L

    """
    def __init__(self, app ): 
        sf = app.repositoryfactory.session_factory
        for cls in app.resources.keys():
            self[cls.__name__] = sf.query(cls)
            print sf.query(cls).first()



class Repo(dict):
    """
           r = Repo(app)
           print r['Simpmtspec'].get( (1,1 ) )   ## CPK get 
    """
    def __init__(self, app ):
        for cls in app.resources.keys():
            repo = app.repositoryfactory( cls )
            self[cls.__name__] = repo
            
            if cls.__name__ == 'Simpmtspec':
            	qf = repo.queryfactory.get( cls )
            	from rum.query import Query, eq 
                
                # should not need to descend to the SA specialization 
                # from rumalchemy.query import SAQuery
            	# saq = SAQuery( eq('ROW', 10 ) , resource=cls ) 
            	# print list( saq.filter( qs['Simpmtspec'] ) )

                q = Query( eq('ROW', 10 ) )
                qq = repo.make_query( request_args=q.as_dict() )
                ss = repo.select(qq)
                print list(ss)

    def query( self , repo , q ): 
        return repo.select( repo.queryfactory( repo.resource , request_args = q.as_dict() ) )




