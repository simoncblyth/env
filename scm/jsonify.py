#!/usr/bin/env python
"""
Convert typical string of javascript options into more valid json
with double quoted word tokens

"""
import re
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
                                data : [],
                                tooltip: {
                                        valueDecimals: 2
                                }
                        }]
                }
'''

def jsonify( js , identity=False ):
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


if __name__ == '__main__':
    ojs = jsonify(js, identity=False)
    print js == ojs
    print ojs

