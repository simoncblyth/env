
Dev notes
-----------

#. Initally had a bug of out of time order series, the resulting drawing caused js timeouts

To manually update from **C2R**, updating the SQLite DB and writing the json files into htdocs/data/scm_backup_check_<node>.json::

    [root@cms02 ~]# env-
    [root@cms02 ~]# scm-backup-
    [root@cms02 ~]# scm-backup-monitor


To update the html docs that present the plots, do a sphinx run. This is not  
not needed every time, as the JSON gets loaded on page load::

   cd $(env-home)
   make                 
   
Check the results:

#. http://localhost/edocs/scm/monitor/
#. http://dayabay.phys.ntu.edu.tw/edocs/scm/monitor/


automated updating
~~~~~~~~~~~~~~~~~~~~~

cronjob on C2R runs the **scm-backup-monitor** with cronline::

   30 19 * * *  ( export HOME=/root ; export NODE=cms02 ; export MAILTO=blyth@hep1.phys.ntu.edu.tw ; export ENV_HOME=/home/blyth/env ; . /home/blyth/env/env.bash ; env-  ; scm-backup- ; scm-backup-monitor ) >  /var/scm/log/scm-backup-monitor-$(date +"\%a").log 2>&1

this doese the fabric run, sqlite persisting and json dumping


highstock and highcharts interference ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots refusing to appear when served from cms02 when the ``templates/layout.html`` contains
**_static/highcharts/highcharts.js** whereas OK locally on G ?


::

	[blyth@cms02 e]$ svn diff  _templates/layout.html
	Index: _templates/layout.html
	===================================================================
	--- _templates/layout.html      (revision 3487)
	+++ _templates/layout.html      (working copy)
	@@ -1,6 +1,6 @@
	{% extends "!layout.html" %}
	 
	-{% set script_files = script_files + ["_static/highstock/highstock.js","_static/highstock/modules/exporting.js", "_static/highcharts/highcharts.js" ] %}
	+{% set script_files = script_files + ["_static/highstock/highstock.js","_static/highstock/modules/exporting.js" ] %}
	 
	{% block rootrellink %}
	     <li><a href="/tracs/env/timeline">env</a> &raquo;</li>


Maybe related to murky practice of building html on G and rsyncing to C2 for presentation rather
than building on C2.



Todo
~~~~~~

#. logging output is mixed up eg ``/var/scm/log/scm-backup-monitor-Thu.log``  : maybe regain the main from **fab** ?
#. currently arbitrarily scaling to improve visibility of disparate valued
#. prepare a separate sphinx for monitoring ?
#. limit checking 
#. send html mail


highstock with jsfiddle
~~~~~~~~~~~~~~~~~~~~~~~~~~

Try out changes interactively

#. http://jsfiddle.net/jswrY/


nginx config
~~~~~~~~~~~~~~~~~

After copying a demo json from C onto WW this is still failing to present at IHEP
with 404 from the below, whereas they work from N

  * `/e/_static/highstock/highstock.js </e/_static/highstock/highstock.js>`_
  * `/e/_static/highstock/modules/exporting.js </e/_static/highstock/modules/exporting.js>`_

Cause turned out to be an extraneous location direction at WW.



serverside highcharts/highstock with nodejs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :google:`node.js highcharts`
* http://blog.davidpadbury.com/2010/10/03/using-nodejs-to-render-js-charts-on-server/
* https://github.com/davidpadbury/node-highcharts
* https://github.com/davidpadbury/node-highcharts/blob/master/lib/node-highcharts.js
* http://stackoverflow.com/questions/8071442/generation-of-svg-on-server-side-using-highcharts
* http://highslide.com/forum/viewtopic.php?f=12&t=16380
* http://nodejs.org/
* https://github.com/tmpvar/jsdom#readme

