
from rumalchemy.util import get_mapper
from rumalchemy.query import SAQuery
from rum.query import Query

class DbiSAQuery(SAQuery):
    @classmethod
    def from_dict(cls, d):
        print "DbiSAQuery.from_dict  overridden %s %s " % ( cls , repr(d ) )
        return SAQuery.from_dict( d )   ## functionality is at Query level but must invoke thru SAQuery to avoid instanciating abstracts 

def dbi_query_override( app ):
    """
         called prior to app finalization ...  works but it feels wrong as 
         are descending to SA level to override something at Query level 
         
               DbiSAQuery < SAQuery < Query 
                                 
    """
    from rum import query as rumquery
    for cls in app.resources.keys():
        rumquery.QueryFactory.register(DbiSAQuery, resource=cls, pred="get_mapper(resource) is not None  ")


