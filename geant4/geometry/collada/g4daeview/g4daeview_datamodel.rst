Datamodel
==========

Organize the plethora of quantities into entities.

Classification considerations.

#. parameters for which changes demand restart of the worker

   * geometry file
   * geometry selection
   * flavor of the worker, non-vbo OR vbo
   * propagation kernel 

#. parameters of kernel launch, easily tuned

   * threads per block
   * max blocks, determines how the work gets split into kernel launches 
   
#. result timings 

   * investigate scaling with number of photons to see 
     what makes sense for comparing different events 



Performance Data Model examples
----------------------------------

MySQL Performance Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fields:

* instrument name
* time start, time end 


* http://dev.mysql.com/doc/refman/5.6/en/events-waits-current-table.html
* http://marcalff.blogspot.tw/2010/01/performance-schema-overview.html




Instrumenting Python Code
----------------------------

* code changes too fast to make line numbers a good may of instrumenting
  better to name waypoints in the code


* http://stackoverflow.com/questions/4691575/how-to-print-source-code-lines-in-python-logger

::

    __file__,
    inspect.getlineno(inspect.currentframe())-1))

Profiling Python
------------------

#. lineprofiler-


Projects for instrumentation 
-----------------------------

* http://www.codeproject.com/Articles/634644/Building-a-code-instrumentation-library-with-Pytho


Graphite : Graphing infrastructure
-----------------------------------

carbon - a Twisted daemon that listens for time-series data
whisper - a simple database library for storing time-series data (similar in design to RRD)
graphite webapp - A Django webapp that renders graphs on-demand using Cairo


* http://graphite.readthedocs.org/en/latest/overview.html

* https://codeascraft.com/2011/02/15/measure-anything-measure-everything/


whisper
~~~~~~~~

* https://github.com/graphite-project/whisper/



