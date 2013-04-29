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

::





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


