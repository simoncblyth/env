#!/usr/bin/env python
"""

* http://lxml.de/parsing.html

::

    In [8]: root.findall('library_geometries')
    Out[8]: []

    In [9]: root.findall('*')                 
    Out[9]: 
    [<Element {http://www.collada.org/2008/03/COLLADASchema}asset at 0x24f2eb8>,
     <Element {http://www.collada.org/2008/03/COLLADASchema}library_effects at 0x24f2ee0>,
     <Element {http://www.collada.org/2008/03/COLLADASchema}library_materials at 0x24f2e68>,
     <Element {http://www.collada.org/2008/03/COLLADASchema}library_geometries at 0x24f2f08>,
     <Element {http://www.collada.org/2008/03/COLLADASchema}library_visual_scenes at 0x24f2f30>,
     <Element {http://www.collada.org/2008/03/COLLADASchema}scene at 0x24f2f58>]

    In [10]: root.findall('{http://www.collada.org/2008/03/COLLADASchema}library_geometries')
    Out[10]: [<Element {http://www.collada.org/2008/03/COLLADASchema}library_geometries at 0x24f2f08>]

"""
from lxml import etree 
tree = etree.parse("../appendix_a.dae")
root = tree.getroot()






