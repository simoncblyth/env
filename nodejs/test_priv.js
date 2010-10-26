
var sys = require('sys'),
  readcfg = require('./priv').readcfg ;

readcfg( "AMQP_LOCAL_CFG" , function(cfg){ sys.puts(sys.inspect( cfg)) ; } ); 


