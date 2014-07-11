#!/usr/bin/env python
"""
"""
import os, logging
log = logging.getLogger(__name__)


def fromjson(path):
    import json
    with open(os.path.expanduser(path),"r") as fp: 
        pd = json.load(fp)
    return dict(map(lambda _:(int(_[0]),str(_[1])),pd.items()))

shortname = lambda name:splitname(name)[1]

def splitname( name, postfix = "Surface"):
    prefix = "__".join(name.split("__")[:-1]) + "__"
    if name.endswith(postfix):
        nam = name[len(prefix):-len(postfix)]
    else: 
        log.debug("name %s not ending with %s" % (name, postfix))
        nam = name[len(prefix):]
    pass
    return prefix, nam



