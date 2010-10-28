= Django Olive : Annotation Live / Live Logger =

Add live updating to the standard comments framework
via integration with backend message queue and socket server.
Allowing IM like interaction in the comments associated
with any object instance. 

== Overview ==

Serverside javascript backend to running websocket server ...
  * node.js(Google V8) + rabbit.js ( node-amqp + socketio )  
  * handling frontend browser connections via websockets  (or fallback?)

= Design / Prototyping =

== Mapping ==

Mapping AMQP exchanges/keys to the domain problem of objects to annotate/comment on  
  * 1 obj-to-annotate <--> 1 exchange 
      * many exchanges (not easy to get list of exchanges)

  * 1 obj-to-annotate <--> 1 routing key
      * one (~few) exchange    

== Similar Projects ==

Chat only using Orbited (no persistence) or rabbitmq via stomp 
  * http://bitbucket.org/nicoechaniz/django-live/src/tip/templates/live/chat.html

== Serverside layout ==

 * AMQP server (rabbitmq) messaging(exchanges and queues) as backend to live updating annotations
 * nginx frontend django + node.js
{{{
But be prepared: nginx don't support http 1.1 while talking to backend so features 
like keep-alive or websockets won't work if you put node behind the nginx.
}}}


=== nginx can't be used as a reverse proxy for node.js if you use websockets ===

 * http://dailyjs.com/2010/03/15/hosting-nodejs-apps/
     * uses nginx '''upstream''' to a different port   

 * http://groups.google.com/group/socket_io/browse_thread/thread/893f2b2ca2fee9a0

{{{
nginx can't be used as a reverse proxy for node.js if you use
websockets. It doesn't support http 1.1 and it is possible it will
never support keep-alive. Moreover, I think that there is no such
reverse proxy that can proxy websocket connections properly at the
moment. Even HAProxy chokes on ver76 standard implemented in latest
browsers and the HAProxy developer says this wouldn't be fixed because
it is ver76 standard that is flawed.

So I think you have to expose your node.js websocket server directly.
}}}

{{{
26 Oct 16:02:39 - socket.io ready - accepting connections
26 Oct 16:02:44 - Initializing client with transport "xhr-multipart"
26 Oct 16:02:44 - Client 07760018738918006 connected
26 Oct 16:02:44 - Initializing client with transport "xhr-multipart"
26 Oct 16:02:44 - Client 4884709098841995 connected
26 Oct 16:02:54 - Client 4884709098841995 disconnected
26 Oct 16:02:54 - Client 07760018738918006 disconnected
}}}

  * '''thusly ... ''' simply expose the socket server and remember to open the port 
     * BUT some claim otherwise

  * google:"nginx websockets proxy_send_timeout proxy_read_timeout"
  * http://wiki.nginx.org/NginxHttpProxyModule



== Issues ==

  * websocket access control / AMQP access control 

  * another mapping issuse ..
     * map SVN/Trac users to rabbitmq ones ???


== Tools ... command line commenter ==

== Monitoring Tools ==

  * google:"rabbitmq munin"
  * http://munin-monitoring.org/ 


== Django Comment Framework integration/customization ==


  * live features ...
     * list of who is connected ?

== Persister ==

 * Need to subscribe to all {{{olive.*}}} exchanges and persist new messages to DB ...
   * as using Django ORM tis natural to do in a management command with pika 
       
   * BUT a list of exchanges is not exposed over the wire 
      * http://groups.google.com/group/rabbitmq-discuss/browse_thread/thread/baafa4f585fb7ba1/204978ec07b444ff
      * workaround via invoking rabbitmqctl on server  ... requires root

      * define one exchange that {{{olive.exchange}}} that contains the names of all exchanges
         * problem here is that need to keep it synced ... TOO COMPLEX ... '''JUST CHANGE MAPPING'''

  * due to these difficulties ... move to single(~few) exchanges 
      * partition for differnt objs by routing key 

=== maybe restructure to avoid  persistence duplication ===

From '''comments.views.comments.post_comment''' a signal 
is called on persisting comments via the normal POST ...

{{{
# Save the comment and signal that it was saved
 comment.save()
 signals.comment_was_posted.send(
     sender  = comment.__class__,
     comment = comment,
     request = request
 )
}}}

  * could use this hook (with pika), to send the JSONified (or even rendered) comment to rabbitmq 
     * sending rendered is good .. it eliminates duplication, simplifying the clientside js 

  * cf '''rabbit.js/pubsub.html''' 
      * would just need to '''sub''' to getting updates 
      *  '''pub''' being done from within the django server on posting 


=== django generic object addressing '''key'''  ===

Names of '''model''' and '''app_label''' gets to the '''ContentType''' , which can then furnish instances 
{{{

  ## fully generic access to an instance given coordinates  (model,app_label,pk) = ("db","table",1)
In [23]: t = ContentType.objects.get(model="Table",app_label="db").get_object_for_this_type( pk=1 )
}}}

   * so generic key '''olive.app_label.model.pk''' for example '''olive.db.table.1''' 
     * (assumes a single django instance ...  just make the '''olive''' prefix configurable to cover this )

=== get the key for annotated object from comment instance ===

{{{
   ".".join([settings.OLIVE_KEY_PREFIX, comment.content_type.app_label, comment.content_type.model, str(comment.content_object.pk)])
}}}

=== access comments for an instance === 

Access comments for an obj ... despite the confusing '''for_model''' name, this works for instances too 
{{{
  Comment.objects.for_model( t )            ## comments for the t instance
  Comment.objects.for_model( t.__class__ )  ## all comments for all table instances 
}}}

=== cutomization of comments ===

  * add/remove fields ...
     * might incorp a rectangle specification : for boxing a figure/image
     * JSON encoding of comments ?

  * tis admin integrated ... so probably easier to just not expose, rather than to remove extraneous fields
     * http://mitchfournier.com/2010/08/12/customizing-django-comments-remove-unwanted-fields/


= STATUS =

   * saved comments are propagated to message queue
   * live updating not working ..
       * socketio looking for bits of self in the wrong server ... 



