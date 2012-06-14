#!/usr/bin/env python
"""
Convert typical string of javascript options into more valid json
with double quoted word tokens
"""
try:
    import json
except ImportError:
    import simplejson as json

import re
def jsonify( js , identity=False ):
    """
    :param js: string containing primitive javascript object 
    :param identity:  when True makes no changes, for testing only

    Matches words like the below and fixes up quoting to allow JSON transport::

          rangeSelector : {       ===>     "rangeSelector" : {


    Although it may be possible to be clever and do this with 
    regexp replacment the below plodding approach has the advantage of extensibility
    and quick development.
    """
    word = re.compile("(\s)(\S+)(\s*)(\:)")
    pos = 0
    ojs = ''
    for m in word.finditer(js):
        start = m.start()
	ojs += js[pos:start]
	beg, end = m.span()
        grps = list(m.groups())
	if not identity:
	    grps[0] = "\""
	    grps[2] = "\""
	rep = "".join(grps)
        ojs += rep
	pos = beg + len(rep)
        if identity:
	    assert js[beg:end] == rep, ("js beg:end mismatch", js[beg:end], rep ) 	
	    assert pos == end, ("pos/end mismatch", pos, end, rep)
    pass
    ojs += js[pos:]    
    if identity:
	assert  js == ojs, "identity failed"
    if not identity:
        ojs = ojs.replace("'","\"")
    return ojs 
    
    
def jsdict( d ):
    """ 	
    :param kwa: context to be json interpolated into the json
    """
    jsd = {}
    if d:
	 for k,v in d.items(): 
            

            jsd[k] = json.dumps( v )
    return jsd


if __name__ == '__main__':
    js = r'''
             {
                        chart : {
                                renderTo : 'container'
                        },

                        rangeSelector : {
                                selected : 1
                        },

                        title : {
                                text : 'AAPL Stock Price'
                        },
                        
                        series : [{
                                name : 'AAPL',
                                data : %(data)s,
                                tooltip: {
                                        valueDecimals: 2
                                }
                        }]
                }
    '''
    ojs = jsonify(js, identity=False )
    print js == ojs
    print ojs

