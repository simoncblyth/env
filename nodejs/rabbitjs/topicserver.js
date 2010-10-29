// based on socketio.js from rabbit.js
var http = require('http');
var url = require('url');
var fs = require('fs');
var io = require('../socket.io-node/lib/socket.io');
var sys = require('sys');
var topic = require('./topic.js');

var server = http.createServer(function (req, res) {
	var path = url.parse(req.url).pathname;
	switch (path){
	case '/':
            path = '/index.html';
	case '/index.html':
        case '/pubsub.html':
        case '/request.html':
        case '/reply.html':
	    fs.readFile(__dirname + path, function(err, data){
		if (err) return send404(res);
		res.writeHead(200, {'Content-Type': 'text/html'});
		res.write(data, 'utf8');
		res.end();
	    });
	    break;
	default: send404(res);
	}
});

send404 = function(res){
	res.writeHead(404);
	res.write('404');
	res.end();
};

var socketserver = io.listen(server);
topic.listen(socketserver);

server.listen(8080);
