
  <script src="http://{{ OLIVE_SERVER_HOST }}:{{ OLIVE_SERVER_PORT }}/socket.io/socket.io.js"></script>
  <script>
      var olive = new io.Socket("{{ OLIVE_SERVER_HOST }}", { port:{{ OLIVE_SERVER_PORT }} } );
      olive.connect();
      olive.send('topic {{ olive_address }}'); 
  </script>

  <!--script>
      olive.on('message', 
          function(msg){
              var el = document.createElement('p');
              el.innerHTML = '<em>' + msg + '</em>';
              document.getElementById('comments').appendChild(el);
              document.getElementById('comments').scrollTop = 1000000;
          }
       );
  </script-->


