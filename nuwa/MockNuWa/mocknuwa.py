#!/usr/bin/env python

import json, pprint, collections

def load(path):
    return json.load(file(path),object_pairs_hook=collections.OrderedDict)







def fmt(k,v):
    try:
        v = int(v)     
    except ValueError:
        return v
     
    if v > 1e6:
        return "%10.2fM " % (float(v)/1e6)         
    else:
        return v




if __name__ == '__main__':
    d = load("/tmp/mocknuwa.json")
    g = d['geometry']

    used = []

    for k,v in g.items():
        extra = ""
        if k.endswith('gpu_used'):
            used.append(v)
            if len(used) > 1:   
                extra = " %s" % (fmt(k,v-used[-2])) 
            pass
        elif k.endswith('count'):
            count = g[k]
            itemsize = g.get(k.replace('count','itemsize'),None) 
            if not itemsize is None: 
                total = count*itemsize
                extra = "  %d * %d = %d  (%10.2f M) " % (itemsize, count, total, float(total)/1.0e6 )
     
        print " %30s :   %s   %s " % (k,fmt(k,v), extra) 





