

Plotter = function ( plot_id, data_url , data_kw , opts ){
   
   this.plot_id = plot_id 
   this.data_url = data_url
   this.data_kw = data_kw
   this.opts = opts
   this.data = []    
 
   this.zero = function(){
       var series = this.opts.series
       for( is = 0 ; is < series.length ; is++ ){
           this.data[is] = []
       }
   }    
 
   this.configure = function(){  //  examine the labels to determine which rows to put into the data 
       this.zero()
       var series = this.opts.series
       for( is = 0 ; is < series.length ; is++ ){
            var label = series[is].label 
            var n     = label.indexOf(':') 
            if ( n > -1 ){
                series[is]._x = label.substring(0,n)
                series[is]._y = label.substring(n+1)
            }
        }
   }
     
   this.handle_data = function(json){  // invoked by load_data when the data arrives 

      var series = this.opts.series 
      for( _i = 0 ; _i < json.items.length ; _i++ ){
      	  item = json.items[_i];
       	  for( is = 0 ; is < series.length ; is++ ){
              var s = series[is]
              this.data[is].push( [item[s._x], item[s._y]] )
          }
      } 	  
      this.plot = $.jqplot( this.plot_id , this.data , this.opts )
   }
   
   this.load_data = function(){    // NB getJSON returns asynchronously
      var obj = this ;   
      $.getJSON( this.data_url , this.data_kw ,  function (json){ 
              obj.handle_data(json);
          })
   }

}


Plotter.setup = function(params){
    
    function param_default(pname,def){
          if(typeof params[pname]=="undefined"){
               params[pname]=def;
           }
    };
    
    param_default( "plot_id"  , "plotid" )
    param_default( "data_url" , "http://localhost:8080/SimPmtSpecDbis.json?")
    param_default( "data_kw"  , "" )
    param_default( "opts"     , {'title':"Default Plot Title"} )
    
    plotr = new Plotter( params["plot_id"], params["data_url"], params["data_kw"], params["opts"] )
    plotr.configure()
    plotr.load_data()
    return plotr 
    
} 