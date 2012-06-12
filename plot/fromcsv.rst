

Plotting from a CSV
=====================

http://www.highcharts.com/documentation/how-to-use

To get this to work I planted same domain data at `data.csv </data/data.csv>`_ which the 
script ajax-gets and parses as it builds the data series to pass to highcharts.



.. raw:: html

    <script type="text/javascript" >

        var options = {
            chart: {
                renderTo: 'container',
                defaultSeriesType: 'column'
            },
            title: {
                text: 'Fruit Consumption'
            },
            xAxis: {
                categories: []
            },
            yAxis: {
                title: {
                    text: 'Units'
                }
            },
            series: []
        };


        $.get('/data/data.csv', function(data) {
            // Split the lines
            var lines = data.split('\n');
            
            // Iterate over the lines and add categories or series
            $.each(lines, function(lineNo, line) {
                var items = line.split(',');
                
                // header line containes categories
                if (lineNo == 0) {
                    $.each(items, function(itemNo, item) {
                        if (itemNo > 0) options.xAxis.categories.push(item);
                    });
                }
                
                // the rest of the lines contain data with their name in the first position
                else {
                    var series = {
                        data: []
                    };
                    $.each(items, function(itemNo, item) {
                        if (itemNo == 0) {
                            series.name = item;
                        } else {
                            series.data.push(parseFloat(item));
                        }
                    });
                    
                    options.series.push(series);
            
                }
                
            });
            
            // Create the chart
            var chart = new Highcharts.Chart(options);
        });


    </script>



In addition to the script a html div is needed (can be done together) with id matching that used in script setup:

.. raw:: html

    <div id="container" style="height: 500px; min-width: 500px"></div>



