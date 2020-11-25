"""





Nikolay Novik’s “Building Apps with Asyncio” slides, presented at PyCon UA 2016. 
The information is dense, but a lot of practical experience is captured in these slides.

https://jettify.github.io/pyconua2016/#/17

Referenced from 

https://www.oreilly.com/library/view/using-asyncio-in/9781492075325/ch04.html


"""
import asyncio, functools, logging
log = logging.getLogger(__name__)
from threading import Thread, Event

logging.basicConfig(
    level=logging.DEBUG,
    format='(%(threadName)-10s) %(message)s',
)

class AioThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop = None
        self.event = Event()

    def run(self):
        """
        set the event once the thread starts
        """
        log.info("[run")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        log.info("[call_soon")
        self.loop.call_soon(self.event.set)
        log.info("]call_soon")
        self.loop.run_forever()
        log.info("]run")

    def add_task(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, loop=self.loop)
        return fut

    def finalize(self, timeout=None):
        self.loop.call_soon_threadsafe(self._loop.stop)
        self.join(timeout=timeout)


def main():
    log.info("[main")
    aiothread = AioThread()
    log.info("[start")
    aiothread.start()
    log.info("]start")
    aiothread.event.wait()

    loop = aiothread.loop
    coro = asyncio.sleep(1, loop=loop)
    future = aiothread.add_task(coro)
    timeout = 2.
    try:
        result = future.result(timeout)
    except asyncio.TimeoutError:
        print('The coroutine took too long, cancelling the task')
        future.cancel()
    except Exception as exc:
        print('The coroutine raised an exception: {!r}'.format(exc))


if __name__ == '__main__':
    main()


