

from rum.query import *
from formencode import variabledecode

from vdbi.dyb import ctx

import logging
log = logging.getLogger(__name__)


class DbiQueryFactory(QueryFactory):
 
    def __call__(self, resource, request_args=None, **kw):
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args, **kw ) 
         print "DbiQueryFactory __call__ override puts the query at my mercy  self:%s query:%s req:%s " % ( self , repr(query) , repr(request_args) )
         
         d = {}
         if request_args:
             d = variabledecode.variable_decode(request_args)
         
         if 'ctx' in d:
             v = d['ctx']
             if 'SimFlag' in v and 'Site' in v and 'DetectorId' in v and 'Timestamp' in v:
                 expr =  and_([ 
                           eq(ctx['Site.attr'],v['Site']) ,
                           eq(ctx['SimFlag.attr'],v['SimFlag']) ,
                           eq(ctx['DetectorId.attr'],v['DetectorId']) ,    
                           lt(ctx['TimeStart.attr'],v['Timestamp']),
                           gt(ctx['TimeEnd.attr'],v['Timestamp']),
                                 ])
                                 
                 ## query = query.add_criteria( expr )  ## this misses the and bracket causing #236 
                 
                 if query.expr:
                     query = query.clone( expr = and_( [query.expr, expr ]) )  
                 else:
                     query = query.clone( expr = expr ) 
                 print "modified query %s" % repr(query) 
             else:
                 log.error("incomplete ctx query")
         else:
             log.debug("no ctx query")
         
                 
         return query 



    
 
    