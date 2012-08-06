Backups of repos and tracs at NTU
-------------------------------------


C : cms01 (checking on the C backup of C2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 * `scm_backup_monitor_C.json </data/scm_backup_monitor_C.json>`_

.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_C.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_C" style="height: 500px; min-width: 500px"></div>



After copying a demo json from C onto WW this is still failing to present at IHEP
with 404 from the below, whereas they work from N

  * `/e/_static/highstock/highstock.js </e/_static/highstock/highstock.js>`_
  * `/e/_static/highstock/modules/exporting.js </e/_static/highstock/modules/exporting.js>`_

TODO:

  * compare nginx config and error/access logs between N and WW  


H1 : hep1
~~~~~~~~~~

 * `scm_backup_monitor_H1.json </data/scm_backup_monitor_H1.json>`_

.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_monitor_H1.json', function(options) {
		window.chart = new Highcharts.StockChart(options);
	});
    });
    </script>
    <div id="container_H1" style="height: 500px; min-width: 500px"></div>



