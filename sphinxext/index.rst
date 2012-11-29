Sphinx Extensions
===================

taglist
----------

Accessing file level docinfo/metadata with **taglist** extension.


.. taglist::


htmlrole
----------

Raw html style and roles are included with::

   .. include:: /sphinxext/roles.txt

Note that absolute include paths are actually relative to sphinx source directory.

.. include:: /sphinxext/roles.txt

For this page simply including **roles.txt** would work. That include is invisible, so a literalinclude
to allow to see the source.

.. literalinclude:: roles.txt

Here is an example of using the roles :alarm:`some alarming text` and :warn:`some warning text` and :ok:`some ok text` and back to normal

The RST of the above line is::

    Here is an example of using the roles :alarm:`some alarming text` and :warn:`some warning text` and :ok:`some ok text` and back to normal


stockchart
------------

.. stockchart:: /data/scm_backup_monitor_C.json container_C


