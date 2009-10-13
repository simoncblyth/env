
from rum.query import Query
from rum import app


def data_url(d , format="json"):
    """ based on CSVLink.update_params 
        replacing    req.host_url + req.path_info + ".json?" + req.query_string   ## http://pythonpaste.org/webob/reference.html  
    
     """
    routes=app.request.routes
    kwds=dict()
    kwds["format"]=format
    if routes['resource'] is not None:
        kwds["resource"]=routes['resource']
    if isinstance(d['value'], Query):
        q = d['value']
        nq = q.clone(limit=None, offset=None)
        kwds.update(nq.as_flat_dict())
    else:
        print "json_url no q %s " % repr(d)
    url = app.url_for(**kwds)
    return url



def json_url(d):return data_url(d,"json")
def csv_url(d):return data_url(d,"csv")
