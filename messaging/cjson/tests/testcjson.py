#!/usr/bin/env python
import json

if __name__ == '__main__':
    d = {}
    d['propagation'] = dict(a=1,b=2.2,c="hello")

    print json.dumps(d)
