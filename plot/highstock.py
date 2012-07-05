from env.plot.jsonify import jsonify, jsdict


class HSOptions(dict):
    """
    Dict containing options to be converted to JSON and passed
    to javascript function via jQuery ajax call::

	window.chart = new Highcharts.StockChart(options);

    """
    js = jsonify(r'''
             {
                        chart : {
                                renderTo : %(renderTo)s
                        },

                        rangeSelector : {
                                selected : 1
                        },

                        xAxis: {
	    	                min: %(xmin)s,
			    	max: %(xmax)s,
				ordinal: false,
				startOnTick: false,
				endOnTick: false	
				},

                        yAxis: [
			        {
		                   title: {
					      text: 'TGZ MB'
					},
				   height: 200,
				lineWidth: 2
				}, 
				{
				   title: {
				             text: 'OK'
				 	},
				     top: 300,
	                          height: 100,
			          offset: 0,
			       lineWidth: 2
			      }
			      ],


                        title : {
                                text : %(title)s
                        },
                        
                        series : %(series)s
             }
	     ''')
    def __repr__(self):
	return self.js % jsdict(self)    
    



