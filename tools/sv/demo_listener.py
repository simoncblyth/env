#!/usr/bin/env python
"""
Supervisord Event Listeners
==============================

http://supervisord.org/events.html
http://supervisord.org/events.html#process-log-event-type

An event listener implementation is a program that is willing to accept
structured input on its stdin stream and produce structured output on its
stdout stream. An event listener implementation should operate in unbuffered
mode or should flush its stdout every time it needs to communicate back to the
supervisord process. Event listeners can be written to be long-running or may
exit after a single request (depending on the implementation and the
autorestart parameter in the eventlistener configuration).

An event listener can send arbitrary output to its stderr, which will be logged
or ignored by supervisord depending on the stderr-related logfile configuration
in its [eventlistener:x] section.


Conf
-----

Config in `supervisord.conf`::

	[eventlistener:demo_listener]
	command=/home/blyth/env/tools/sv/demo_listener.py
	events=PROCESS_STATE,TICK_60,PROCESS_LOG_STDOUT,PROCESS_LOG_STDERR

Debugging Tips
----------------

#. having multiple supervisorctl open and tailing various processes is useful to check the listener is hearing about activity wrt other processes


Understanding Operation
------------------------

To follow the debug ouput on stderr (do not write to stdout as that will confuse communication with supervisord)::

	N> tail -f demo_listener stderr
	==> Press Ctrl-C to exit <==
	TATE_STOPPING len:90
	 processname:demo_event_listener groupname:demo_event_listener from_state:RUNNING pid:29391header line: ver:3.0 server:supervisor serial:15 pool:demo_event_listener poolserial:15 eventname:PROCESS_STATE_STOPPED len:91
	 processname:demo_event_listener groupname:demo_event_listener from_state:STOPPING pid:29391header line: ver:3.0 server:supervisor serial:16 pool:demo_event_listener poolserial:16 eventname:PROCESS_STATE_STARTING len:88
	 processname:demo_event_listener groupname:demo_event_listener from_state:STOPPED tries:0header line: ver:3.0 server:supervisor serial:17 pool:demo_event_listener poolserial:17 eventname:PROCESS_STATE_RUNNING len:90
	 processname:demo_event_listener groupname:demo_event_listener from_state:STARTING pid:3949header line: ver:3.0 server:supervisor serial:18 pool:demo_event_listener poolserial:18 eventname:TICK_60 len:15
	 when:1366960740header line: ver:3.0 server:supervisor serial:19 pool:demo_event_listener poolserial:19 eventname:TICK_60 len:15
	 when:1366960800header line: ver:3.0 server:supervisor serial:20 pool:demo_event_listener poolserial:20 eventname:TICK_60 len:15
	 when:1366960860

For a change to this script to take effect need to restart it::

	N> restart demo_listener
	demo_listener: stopped
	demo_listener: started
	N> tail -f demo_listener stderr
        ...


Process state changes 
-----------------------

The result of stopping and starting mysql::

	header[ver:3.0 server:supervisor serial:67 pool:demo_event_listener poolserial:67 eventname:PROCESS_STATE_STOPPING len:61]
	payload[processname:mysql groupname:mysql from_state:RUNNING pid:2997]
	header[ver:3.0 server:supervisor serial:68 pool:demo_event_listener poolserial:68 eventname:PROCESS_STATE_STOPPED len:62]
	payload[processname:mysql groupname:mysql from_state:STOPPING pid:2997]
	header[ver:3.0 server:supervisor serial:69 pool:demo_event_listener poolserial:69 eventname:PROCESS_STATE_STARTING len:60]
	payload[processname:mysql groupname:mysql from_state:STOPPED tries:0]
	header[ver:3.0 server:supervisor serial:70 pool:demo_event_listener poolserial:70 eventname:PROCESS_STATE_RUNNING len:63]
	payload[processname:mysql groupname:mysql from_state:STARTING pid:24892]    


Listening for log updating 
----------------------------

Despite adding the below to the dybslv config so far did not receive log updating events::

	[program:dybslv]
	stdout_events_enabled=true
	stderr_events_enabled=true

In order to make dybslv audible with added config above had to `stop`, `remove` then `add` the dybslv.


event buffer overflowed
-------------------------

* http://supervisord.org/events.html

A listener pool has an event buffer queue. The queue is sized via the listener
pools `buffer_size` config file option. If the queue is full and supervisor
attempts to buffer an event, supervisor will throw away the oldest event in the
buffer and log an error.

Maybe my listener is not properly expunging events as they appear to be handled but show
up in the discarded::


    N> tail -f demo_listener stderr
    ==> Press Ctrl-C to exit <==
    2013-05-10 20:44:58,289 __main__ INFO     handling event PROCESS_LOG_STDOUT 
    2013-05-10 20:44:58,289 __main__ INFO     headers  : [{'ver': '3.0', 'poolserial': '103593', 'len': '139', 'server': 'supervisor', 'eventname': 'PROCESS_LOG_STDOUT', 'serial': '103758', 'pool': 'demo_listener'}]
    2013-05-10 20:44:58,290 __main__ INFO     pheaders : [{'processname': 'demo_logger', 'pid': '31611', 'channel': 'stdout', 'groupname': 'demo_logger'}]
    2013-05-10 20:44:58,290 __main__ INFO     pdata    : [2013-05-10 20:44:58,289 __main__ INFO     hi using index 0 delay 3]
    2013-05-10 20:45:00,293 __main__ INFO     handling event TICK_60 
    2013-05-10 20:45:00,293 __main__ INFO     headers  : [{'ver': '3.0', 'poolserial': '103594', 'len': '15', 'server': 'supervisor', 'eventname': 'TICK_60', 'serial': '103759', 'pool': 'demo_listener'}]
    2013-05-10 20:45:00,293 __main__ INFO     payload  : [when:1368189900]
    2013-05-10 20:45:01,290 __main__ INFO     handling event PROCESS_LOG_STDOUT 
    2013-05-10 20:45:01,290 __main__ INFO     headers  : [{'ver': '3.0', 'poolserial': '103595', 'len': '139', 'server': 'supervisor', 'eventname': 'PROCESS_LOG_STDOUT', 'serial': '103760', 'pool': 'demo_listener'}]
    2013-05-10 20:45:01,290 __main__ INFO     pheaders : [{'processname': 'demo_logger', 'pid': '31611', 'channel': 'stdout', 'groupname': 'demo_logger'}]
    2013-05-10 20:45:01,290 __main__ INFO     pdata    : [2013-05-10 20:45:01,289 __main__ INFO     hi using index 0 delay 3]

    N> maintail
    2013-05-10 20:44:49,288 ERRO pool demo_listener event buffer overflowed, discarding event 103745
    2013-05-10 20:44:52,289 ERRO pool demo_listener event buffer overflowed, discarding event 103746
    2013-05-10 20:44:55,289 ERRO pool demo_listener event buffer overflowed, discarding event 103747
    2013-05-10 20:44:58,289 ERRO pool demo_listener event buffer overflowed, discarding event 103748
    2013-05-10 20:45:00,292 ERRO pool demo_event_listener event buffer overflowed, discarding event 103543
    2013-05-10 20:45:00,292 ERRO pool demo_listener event buffer overflowed, discarding event 103749
    2013-05-10 20:45:01,290 ERRO pool demo_listener event buffer overflowed, discarding event 103750
    2013-05-10 20:45:04,290 ERRO pool demo_listener event buffer overflowed, discarding event 103751
    2013-05-10 20:45:07,291 ERRO pool demo_listener event buffer overflowed, discarding event 103752


Changing logging delay from 3 to 10 s, changed the discard pulse accordingly::

    2013-05-10 20:51:00,275 ERRO pool demo_event_listener event buffer overflowed, discarding event 103672
    2013-05-10 20:51:00,275 ERRO pool demo_listener event buffer overflowed, discarding event 103874
    2013-05-10 20:51:04,274 ERRO pool demo_listener event buffer overflowed, discarding event 103875
    2013-05-10 20:51:14,274 ERRO pool demo_listener event buffer overflowed, discarding event 103876
    2013-05-10 20:51:24,274 ERRO pool demo_listener event buffer overflowed, discarding event 103877
    2013-05-10 20:51:34,274 ERRO pool demo_listener event buffer overflowed, discarding event 103878


"""
import sys, logging
log = logging.getLogger(__name__)
from supervisor import childutils

class LogListener(object):
    def __init__(self):
        self.stdin = sys.stdin 
        self.stdout = sys.stdout 
        self.stderr = sys.stderr
        self.handlers = dict(PROCESS_LOG=self.handle_process_log , TICK=self.handle_tick )
        self.logupdate = {} 
        pass

    def dispatch(self, headers, payload):
        for prefix,handler in self.handlers.items():
            if headers['eventname'].startswith(prefix):
                log.info("handling event %s " % headers['eventname'] )
                handler( headers, payload) 
                return True
        return False

    def __call__(self):
        while 1:
            headers, payload = childutils.listener.wait(self.stdin, self.stdout)
            if not self.dispatch(headers, payload):
                log.info("unhandled event %s " % headers['eventname'] )
                childutils.listener.ok(self.stdout)
                continue        # keep while listening
            pass
            self.stderr.flush()
            childutils.listener.ok(self.stdout)

    def handle_process_log(self, headers, payload):
         """
         :param header: generic headers
         :param payload: payload
         """
         pheaders, pdata = childutils.eventdata(payload)
         log.info("headers  : [%s]" % headers )
         log.info("pheaders : [%s]" % pheaders )
         log.info("pdata    : [%s]" % pdata.strip() )
         assert headers['eventname'].startswith('PROCESS_LOG')

    def handle_tick(self, headers, payload):
         """
         :param header: generic headers
         :param payload:
         """
         log.info("headers  : [%s]" % headers )
         log.info("payload  : [%s]" % payload.strip() )
         assert headers['eventname'].startswith('TICK')


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s")
    LogListener()()

if __name__ == '__main__':
    main()


