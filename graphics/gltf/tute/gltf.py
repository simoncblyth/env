#!/usr/bin/env python

"""

* https://github.com/jzitelli/python-gltf-experiments


In [22]: g.keys()
Out[22]: 
[u'accessors',
 u'scenes',
 u'meshes',
 u'nodes',
 u'bufferViews',
 u'buffers',
 u'asset']

import base64

assert base64.b64decode(res.json()['data'][len('data:application/octet-stream;base64,'):]) == data


"""

import os, logging, json, base64


json_ = lambda path:json.load(file(os.path.expandvars(os.path.expanduser(path))))


if __name__ == '__main__':
    g = json_("minimal.gltf")
    version = g['asset']['version']

    aa = g['accessors']
    ss = g['scenes']
    mm = g['meshes']
    nn = g['nodes']
    vv = g['bufferViews']
    bb = g['buffers']
    ii = g['asset']


    b = bb[0]     
    data = base64.b64decode( b['uri'][len('data:application/octet-stream;base64,'):] )
    assert len(data) == b['byteLength']


    s = ss[0]

    nn = s['nodes']

    n = nn[0]

    m = mm[n]

    p = m['primitives'][0]





