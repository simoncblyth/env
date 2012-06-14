Monitor
=========

And using a raw html directive to embed the javascript(use show source on right to see this) that accesses the data and configures the plot:

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


.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_C.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_C" style="height: 500px; min-width: 500px"></div>


.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_H1.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_H1" style="height: 500px; min-width: 500px"></div>








