Collada Intro
======================

COLLADA defines an XML Namespace and database schema to make it easy to
transport 3D assets between applications without loss of information, enabling
diverse 3D authoring and processing tools to be combined into a content
production pipeline.


References
----------

* http://www.khronos.org/collada/

Collada XSD Schema
--------------------

More readable than you might expect. See precisely which elements are allowed where.

* http://www.khronos.org/files/collada_schema_1_5
* http://www.khronos.org/files/collada_reference_card_1_4.pdf

::

    curl -L http://www.khronos.org/files/collada_schema_1_5 > collada_schema_1_5.xsd

Questions
----------

* :google:`collada mesh vertices triangles example`
* :google:`collada double sided surfaces`
* :google:`collada node instance_node distinction`





pycollada
-----------

Based on lxml and numpy, so should be fast enough.

* http://pycollada.github.io/index.html
* http://pycollada.github.io/structure.html

  * looks to be simple to convert collada into STL or get chroma to use pycollada
  * Each vertex has 2 texture coordinates. 
  * Could encode material index into texture coordinates OR equivalently have a material texture map.




