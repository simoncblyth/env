
from carrot.connection import DjangoBrokerConnection
from carrot.messaging import Publisher, Consumer
from abtruninfo.models import AbtRunInfo
import cjson

def send_dummy_runinfo(run_info):
    """Send a dummy message for adding a runinfo entry ."""
    connection = DjangoBrokerConnection()
    publisher = Publisher(connection=connection,
                          exchange="runinfo_run",
                          routing_key="runinfo_run_add",
                          exchange_type="direct")

    publisher.send(run_info)

    publisher.close()
    connection.close()


def datetimes_( d ):
    from datetime import datetime
    u = {}
    for k,v in d.items():
        if k.endswith("Time"):
            u[k] = datetime.strptime( v , "%c" )
    d.update( u )
    return d

def skips_(d): 
    for k, v in d.items():
        if type(v) == str and v.startswith("SKIPPED"):del d[k]
    return d      

def deserialize( body ):
    d = None
    try:
        d = cjson.decode( body )
    except:
        print "failed to decode %s " % body
    return d 
     
def process_abtruninfo():
    """Process all currently gathered runinfo by saving them to the
    database."""
    connection = DjangoBrokerConnection()
    consumer = Consumer(connection=connection,
                        queue="runinfo_abtruninfo",
                        exchange="runinfo_abtruninfo",
                        routing_key="runinfo_abtruninfo_add",
                        exchange_type="direct")

    for message in consumer.iterqueue():
        d = deserialize( message.body )
        d = datetimes_(d)
        d = skips_(d)
        print message, d 
        if d:
            ari = AbtRunInfo( **d )
            print ari 
        #message.ack()

    #  usual django db updating 
    #for url, click_count in clicks_for_urls.items():
    #    Run.objects.increment_clicks(url, click_count)
    #    # Now that the clicks has been registered for this URL we can
    #    # acknowledge the messages
    #    [message.ack() for message in messages_for_url[url]]

    consumer.close()
    connection.close()



