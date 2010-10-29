// based on sockets.js from rabbit.js 
var amqp = require('../node-amqp/');
var sys = require('sys');

var connection ;
var readcfg = require( process.env["ENV_HOME"] + '/nodejs/priv').readcfg ;
readcfg(  "AMQP_CFG" ,  
   function( cfg ){ 
       sys.puts("Configuring AMQP connection ...");
       sys.puts(sys.inspect( cfg ))
       connection = amqp.createConnection(cfg) ; 
   } ); 

var debug = (process.env['DEBUG']) ?
    function(msg) { sys.debug(msg) } : function() {};

function topicSocket(client, addr ) {
    var i = addr.indexOf(':');
    var exchangeName = (i > -1) ? addr.substring(0, i) : addr ;
    var routingKey   = (i > -1) ? addr.substr(i+1) : '#.string' ;
    
    sys.log('sub socket opened : addr ' + addr + ' exchangeName : ' + exchangeName + ' routingKey : ' + routingKey  );
  
    var exchange = connection.exchange(exchangeName, {'type': 'topic'});
    var queue = connection.queue('');
    queue.subscribe(function(message) {
        debug('topic:'); debug(message);
        client.send(message.data);
    });
    queue.bind(exchange.name, routingKey );
}

function listen(server, allowed) {
    server.on('connection', function (client) {
        function dispatch(msg) {
            client.removeListener('message', dispatch);
            msg = msg.toString();
            var i = msg.indexOf(' ');
            var type = (i > -1) ? msg.substring(0, i) : msg;
            var addr = (i > -1) ? msg.substr(i+1) : '';
            if (check_rendezvous(type, addr, allowed)) {
                switch (type) {
                case 'topic':
                    topicSocket(client, addr)
                    break;;
                default:
                    client.send("Unknown socket type");
                    client.end();
                    sys.log("Unknown socket type in: " + msg);
                }
            }
            else {
                client.send("Unauthorised rendezvous");
                client.end();
                sys.log("Access denied: " + type + " to " + addr);
            }
        }
        client.on('message', dispatch);
    });
}

function check_rendezvous(type, addr, allowed) {
    if( type == 'topic' ) return true;  // could restrict AMQP addresses here (ie exchange:routingKey)  
    if (!allowed) return true; // no explicit list = everything goes
    var socks = allowed[addr];
    return socks && socks.indexOf(type) > -1
}

exports.listen = listen;
