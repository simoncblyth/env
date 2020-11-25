"""

https://jettify.github.io/pyconua2016/#/21

https://github.com/aio-libs/janus


"""
import asyncio
import janus
import logging 
log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format='(%(threadName)-10s) %(message)s',
)

def threaded(sync_q):
    for i in range(100):
        log.info(i) 
        sync_q.put(i)
        log.info(i) 
    sync_q.join()

async def async_coro(async_q):
    for i in range(100):
        val = await async_q.get()
        assert val == i
        log.info(i) 
        async_q.task_done()
        log.info(i) 

async def main():
    queue = janus.Queue()
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(None, threaded, queue.sync_q)
    await async_coro(queue.async_q)
    await fut
    queue.close()
    await queue.wait_closed()


asyncio.run(main())

