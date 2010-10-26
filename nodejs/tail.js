/*

   Live tailing of file 

   http://blog.new-bamboo.co.uk/2009/12/7/real-time-online-activity-monitor-example-with-node-js-and-websocket

   [blyth@belle7 nodejs]$ node tail.js $HOME/dummy.txt
   [blyth@belle7 nodejs]$ echo hello >> ~/dummy.txt 


*/
  var sys = require('sys')
  var spawn = require('child_process').spawn ;
  var filename = process.ARGV[2];

  if (!filename)
    return sys.puts("Usage: node tail.js filename");

  // Look at http://nodejs.org/api.html#_child_processes for detail.
  var tail = spawn("tail", ["-f", filename]);
  sys.puts("start tailing");

  tail.stdout.on("data", function (data) {
    sys.puts(data);
  });

  // From nodejs.org/jsconf.pdf slide 56
  var http = require("http");
  http.createServer(function(req,res){
    res.writeHead(200,{"Content-Type": "text/plain"});
    tail.stdout.on("data", function (data) {
      res.write(data);
    });  
  }).listen(8000);



