#!/usr/bin/env python
"""
Dump docutils node tree of an RST document. 

* http://stackoverflow.com/questions/20793240/how-to-print-a-restructuredtext-node-tree

Usage::

    simon:~ blyth$ doctree.py /tmp/ntu-report-may-2017.rst 
    <document source="<string>">
        <docinfo>
            <field>
                <field_name>
                    title
                <field_body>
                    <paragraph>
                        Optical Photon Simulation on GPUs
            <date>
                May 1, 2017
        <section ids="simon-c-blyth" names="simon\ c\ blyth">
            <title>
    ...
        

"""
import sys
from docutils.core import publish_string

if __name__ == '__main__':
     path = sys.argv[1]
     rst = file(path).read()
     print publish_string(rst)



