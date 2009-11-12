
from carrot.connection import DjangoBrokerConnection
from carrot.messaging import Publisher, Consumer
from run.models import Run


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


def process_runinfo():
    """Process all currently gathered runinfo by saving them to the
    database."""
    connection = DjangoBrokerConnection()
    consumer = Consumer(connection=connection,
                        queue="runinfo_run",
                        exchange="runinfo_run",
                        routing_key="runinfo_run_add",
                        exchange_type="direct")

    for message in consumer.iterqueue():
        print message, message.body
        message.ack()

    #  usual django db updating 
    #for url, click_count in clicks_for_urls.items():
    #    Run.objects.increment_clicks(url, click_count)
    #    # Now that the clicks has been registered for this URL we can
    #    # acknowledge the messages
    #    [message.ack() for message in messages_for_url[url]]

    consumer.close()
    connection.close()



