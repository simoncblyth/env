#!/usr/bin/env python
"""
::

    In [10]: for iss in js['issues']:print len(iss),iss['content_updated_on']
    16 1970-01-01T08:20:46.008958+00:00
    16 1970-01-01T08:20:46.009005+00:00
    16 1970-01-01T08:20:46.008903+00:00
    16 1970-01-01T08:20:46.009044+00:00

"""
import json

def readjson(path):
    with open(path,"rb") as fp:
        js = json.load(fp)
    return js

def main():
    js = readjson("/tmp/t/env.json")
    import IPython
    IPython.embed()  


if __name__ == '__main__':
    main()

