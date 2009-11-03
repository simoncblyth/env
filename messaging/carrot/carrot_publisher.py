
from carrot_connection import conn, params
from carrot.messaging import Publisher
publisher = Publisher(connection=conn, exchange="feed", routing_key="importer")

print "publishing to %s " % repr(params)
publisher.send({"import_feed": "http://cnn.com/rss/edition.rss"})
publisher.close()




