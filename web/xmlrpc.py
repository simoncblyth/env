#!/usr/bin/env python
"""
./xmlrpc.py g4pb_trac -l debug

"""
from cnf import cnf_
from autobrowser import AutoBrowser
from lxml.etree import tostring

if __name__ == '__main__':
    cnf = cnf_(__doc__)
    br = AutoBrowser(cnf)
    #for url in cnf['targets'].split():
    #    tree = br.open_(url, parse=True)
    #    print url, tree, tostring(tree)

    for url in cnf['xmltargets'].split():
        tree = br.xmlopen_(url)
        print url, tree, tostring(tree)




