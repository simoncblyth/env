

Plotter = function ( plot_id, data_url , data_kw ){
   
   this.plot_id = plot_id 
   this.data_url = data_url
   this.data_kw = data_kw
   
   this.handle_data = function(data){
      var cols = [];
      for( _i = 0 ; _i < data.items.length ; _i++ ){
      	  item = data.items[_i];
       	  cols.push( [item.ROW, item.DARKRATE] );    
       }
       this.plot = $.jqplot( this.plot_id , [cols] , {} )
   }
   
   this.load_data = function(){
      var obj = this ;    // NB this is done async
      $.getJSON( this.data_url , this.data_kw ,  function (data){ 
              obj.handle_data(data);
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
    
    plotr = new Plotter( params["plot_id"], params["data_url"], params["data_kw"] )
    plotr.load_data()
    return plotr 
    
} 