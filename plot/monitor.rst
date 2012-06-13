Monitor
=========

And using a raw html directive to embed the javascript(use show source on right to see this) that accesses the data and configures the plot:

#. Initally had a bug of out of time order series, the resulting drawing was horribly slow.

To update::

   ssh C2R
   cd ~blyth/env/scm
   python-
   fab scm_backup_check   # this writes the scm_backup_check.json into the apache htdocs/data dir

   cd ~/blyth/env
   make                   # do a sphinx run


To do:

#. currently are only checking tgzs on a single remote node **C** : display node and expand to all backup nodes
#. split or somehow scale to improve visibility of disparate valued
#. shorten the names
#. prepare a separate sphinx for monitoring 
#. automate the fabric run and sqlite persisting and json dumping
#. send html mail



.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('/data/scm_backup_check.json', function(data) {
		window.chart = new Highcharts.StockChart({
			chart : {
				renderTo : 'container'
			},

			title : {
				text : 'SCM Backup Check'
			},
			
			series : data
                       
		});
	});
    });
    </script>
    <div id="container" style="height: 500px; min-width: 500px"></div>










