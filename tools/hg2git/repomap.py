#!/usr/bin/env python

import sys, json, re
from collections import OrderedDict as odict 


if __name__ == '__main__':

    d = odict()
    for line in sys.stdin.readlines():
        repo = line.strip()
        assert repo.endswith("_hg"), repo
        name = repo[:-3]
        d[repo] = name 
        #print("%s:%s"%(repo,name))
    pass
    print(json.dumps(d, indent=4, sort_keys=False)) 




