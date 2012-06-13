Graphing 
=========

Many js charting/plotting libs
 * http://www.1stwebdesigner.com/css/top-jquery-chart-libraries-interactive-charts/
 

highcharts
-----------

http://www.highcharts.com/documentation/how-to-use

http://www.highcharts.com/download


highcharts in sphinx ?
-------------------------

http://stackoverflow.com/questions/9444342/adding-a-javascript-script-tag-some-place-so-that-it-works-for-every-file-in-sph

http://sphinx.pocoo.org/templating.html#script_files

Grab from ``/usr/local/env/plot/Highstock-1.1.6/examples/basic-line/index.htm``

Avoid having to work out script depth by removing the direct refernces::

    <script src="../../../../_static/js/highstock.js"></script>
    <script src="../../../../_static/js/modules/exporting.js"></script>

Instead reference the js from an added  ``_templates/layout.html``::

        {% extends "!layout.html" %}

        {% set script_files = script_files + ["_static/js/highstock.js","_static/js/modules/exporting.js"] %}

        {% block rootrellink %}
            <li><a href="http://project.invalid/">Project Homepage</a> &raquo;</li>
                {{ super() }}
                {% endblock %}

Where the ``js`` is refering to::

    /Users/blyth/w/_static/js -> /usr/local/env/plot/Highstock-1.1.6/js


And using a raw html directive to embed the javascript(use show source on right to see this) that accesses the data and configures the plot:

.. raw:: html

    <script type="text/javascript" >
    $(function() {
	$.getJSON('http://www.highcharts.com/samples/data/jsonp.php?filename=aapl-c.json&callback=?', function(data) {
		// Create the chart
		window.chart = new Highcharts.StockChart({
			chart : {
				renderTo : 'container'
			},

			rangeSelector : {
				selected : 1
			},

			title : {
				text : 'AAPL Stock Price'
			},
			
			series : [{
				name : 'AAPL',
				data : data,
				tooltip: {
					valueDecimals: 2
				}
			}]
		});
	});

    });
    </script>


.. .. raw:: html

    <script type="text/javascript" >

    var chart1; // globally available
    $(document).ready(function() {
        chart1 = new Highcharts.StockChart(
                     {
                       chart: { renderTo: 'container' }, 
               rangeSelector: { selected: 1 },
                      series: [{
                                 name: 'demo',
                                 data: [[1335830400000,582.13],
                                        [1335916800000,585.98],
                                        [1336003200000,581.82],
                                        [1338422400000,577.73]]
                              }]
                     });
      });

    </script>

In addition to the script a html div is needed (can be done together) with id matching that used in script setup:

.. raw:: html

    <div id="container" style="height: 500px; min-width: 500px"></div>








