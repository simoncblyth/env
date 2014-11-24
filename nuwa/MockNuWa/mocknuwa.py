#!/usr/bin/env python

import json, pprint, collections

def load(path):
    return json.load(file(path),object_pairs_hook=collections.OrderedDict)

def fmt(v):
    return float(v)/1e6 if v>1e6  else v


if __name__ == '__main__':
    d = load("/tmp/mocknuwa.json")
    g = d['geometry']

    for pfx in "a b c d e".split():
        tot = "%s_gpu_total" % pfx
        fre = "%s_gpu_free" % pfx
        g['%s_gpu_used' % pfx] = g[tot] - g[fre]     
        del g[tot]
        del g[fre]



    print "\n".join([" %30s : %10.2fM  %s  " % (k,fmt(v),v) for (k,v) in g.items()])





