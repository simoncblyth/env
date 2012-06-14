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

* can the plot be rendered as an image on the server ? 

   * would allow this to be done once only 



Dev notes
-----------

#. Initally had a bug of out of time order series, the resulting drawing caused js timeouts

To update::

   ssh C2R

   scm-backup-
   scm-backup-monitor     # updates DB and writes the htdocs/data/scm_backup_check_<node>.json 

   cd $(env-home)
   make                   # sphinx run : not needed every time, as the JSON gets loaded on page load
   
   open http://localhost/edocs/scm/monitor/
   open http://dayabay.phys.ntu.edu.tw/edocs/scm/monitor/


To do:

#. currently arbitrarily scaling to improve visibility of disparate valued
#. prepare a separate sphinx for monitoring ?
#. automate the fabric run and sqlite persisting and json dumping
#. send html mail


