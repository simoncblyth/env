Monitor
=========

Backup of repos and tracs from cms02
-------------------------------------

C : cms01
~~~~~~~~~~


.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_C.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_C" style="height: 500px; min-width: 500px"></div>


H1 : hep1
~~~~~~~~~~


.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_H1.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_H1" style="height: 500px; min-width: 500px"></div>


How it works
---------------

Sphinx ``.. raw:: html`` directives are used to embed javascript (use show source on right to see this) and a single **div** into the html built version of this 
page. On page load the javascript runs an ajax query to pull in the plot data and options from a static JSON files for each remote node residing in `</data/>`_. These 
static files are created by the ``scm-backup-monitor`` which using **fabric** to gather info from remote nodes and updates an SQLite DB.


If had large numbers of plots to render, it would be silly to re-render in browser
for quntities that are only updated daily.  But that is what this is doing.  

* can the plot be rendered as an image on the server ? allow this to be done once only 

Doing this on dayabay.ihep.ac.cn ?
------------------------------------

Done:

#. python2.5.6 + sphinx + docutils etc... into  ~/local python
#. fabric + simplejson 
#. caution this will not work in the system python2.3 (used by apache/modpython/trac)
#. nginx running on 8080
#. add env symbolic link to nginx docs
#. hook up the javascript with link in _static::

g4pb-2:~ blyth$ ls -l ~/e/_static/
total 8
lrwxr-xr-x  1 blyth  staff  38 12 Jun 19:45 highstock -> /usr/local/env/plot/Highstock-1.1.6/js
g4pb-2:~ blyth$ 


Hmm the link approach not working with nginx on WW

  * http://dayabay.phys.ntu.edu.tw/e/_static/highstock/highstock.js
  * http://dayabay.ihep.ac.cn:8080/e/_static/highstock/highstock.js




C : cms01 (checking on the C backup of C2, as no SDU access)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_C.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_C" style="height: 500px; min-width: 500px"></div>



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

