from env.plot.jsonify import jsonify, jsdict

class HSSeries(dict):
    """
    Not working 
    """
    js = jsonify(r'''
            {
                  name : %(name)s,
		  data : %(data)s,
                tooltip : {
	                      valueDecimals : 2
			  },
	    }
              ''')
    def __repr__(self):
	return self.js % jsdict(self)    
 

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

                        title : {
                                text : %(title)s
                        },
                        
                        series : %(series)s
             }
	     ''')
    def __repr__(self):
	return self.js % jsdict(self)    
    



