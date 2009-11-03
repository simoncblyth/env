from carrot_connection import conn
from carrot.messaging import Publisher

def publish_binary( path ):
    publisher = Publisher(connection=conn, exchange="feed", routing_key="import_pictures")
    print "publish_binary sending %s " % path 
    publisher.send(open( path ,'rb').read(), content_type="image/jpeg", content_encoding="binary")
    publisher.close()

if __name__=='__main__':
    import sys
    publish_binary( sys.argv[1] )


