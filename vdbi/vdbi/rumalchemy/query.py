
from rumalchemy.util import get_mapper
from rumalchemy.query import SAQuery
from rum.query import Query

from formencode import variabledecode
from formencode.validators import Int


class _sort(_QueryElement):
    @classmethod
    def from_dict(cls, d):
        cls = cls.by_name(d.get('dir', 'asc'))
        return cls(d['c'])


class DbiSAQuery(SAQuery):
    @classmethod
    def from_dict(cls, d):
        print "DbiSAQuery.from_dict  overridden %s %s " % ( cls , repr(d ) )
        ##return SAQuery.from_dict( d )   ## functionality is at Query level but must invoke thru SAQuery to avoid instanciating abstracts 

        expr = sort = limit = offset = None
        #from vdbi import debug_here
        #debug_here()
        d = variabledecode.variable_decode(d)
        if 'q' in d and 'c' in d['q']:
            expr = Expression.from_dict(d['q'])
        if 'sort' in d:
            sort = d['sort']
            if sort:
                sort = map(_sort.from_dict, sort)
        if 'limit' in d:
            limit = Int(min=0).to_python(d['limit'])
        if 'offset' in d:
            offset = Int(min=0).to_python(d['offset'])
        return cls(expr, sort, limit, offset)




def dbi_query_override( app ):
    """
         called prior to app finalization ...  works but it feels wrong as 
         are descending to SA level to override something at Query level 
         
               DbiSAQuery < SAQuery < Query 
                                 
    """
    from rum import query as rumquery
    for cls in app.resources.keys():
        rumquery.QueryFactory.register(DbiSAQuery, resource=cls, pred="get_mapper(resource) is not None  ")


