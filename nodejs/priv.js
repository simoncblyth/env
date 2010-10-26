/*
   starting point bash integration for config access 

      *  NB all strings in JSON must be double quoted 

   groking async ...
        http://gfxmonk.net/2010/07/04/defer-taming-asynchronous-javascript-with-coffeescript.html

          * no-choice , when dealing with async ... are forced to use callback in order 
           to get the return into calling scope

    Usage example :

   var connection ;
   var readcfg = require( process.env["ENV_HOME"] + '/nodejs/priv').readcfg ;
   readcfg(  "AMQP_LOCAL_CFG" ,
      function( cfg ){
          sys.puts("Configuring AMQP connection ...");
          sys.puts(sys.inspect( cfg ))
          connection = amqp.createConnection(cfg) ;
       } );


*/

var   sys = require('sys'),
     exec = require('child_process').exec ;

function readcfg( name , callback ){
   var priv = "priv(){ . "+process.env["ENV_PRIVATE_PATH"]+" ; echo \\$"+name+" ;};priv" ;
   var cmd = "bash -c \"" + priv + "\"" ; 
   var child = exec( cmd , 
         function (error, stdout, stderr) {
            if (error !== null) {
               sys.puts('stdout: ' + stdout);
               sys.puts('stderr: ' + stderr);
               console.log('exec error: ' + error);
            }
            callback( JSON.parse( stdout ) );
         });
 }

exports.readcfg = readcfg





