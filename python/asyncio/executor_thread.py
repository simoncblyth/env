# https://pymotw.com/3/asyncio/executors.html
# asyncio_executor_thread.py
"""

Combining Coroutines with Threads and Processes

A lot of existing libraries are not ready to be used with asyncio natively.
They may block, or depend on concurrency features not available through the
module. It is still possible to use those libraries in an application based on
asyncio by using an executor from concurrent.futures to run the code either in
a separate thread or a separate process.

"""
import asyncio
import concurrent.futures
import logging
import sys
import time


def blocks(n):
    log = logging.getLogger('blocks({})'.format(n))
    log.info('running')
    time.sleep(0.1)
    log.info('done')
    return n ** 2


async def run_blocking_tasks(executor):
    log = logging.getLogger('run_blocking_tasks')
    log.info('starting')
    log.info('creating executor tasks')

    loop = asyncio.get_event_loop()

    blocking_tasks = [ loop.run_in_executor(executor, blocks, i) for i in range(6) ] 

    log.info('waiting for executor tasks')
    completed, pending = await asyncio.wait(blocking_tasks)
    results = [t.result() for t in completed]

    log.info('results: {!r}'.format(results))

    log.info('exiting')


if __name__ == '__main__':
    # Configure logging to show the name of the thread
    # where the log message originates.
    logging.basicConfig(
        level=logging.INFO,
        format='%(threadName)10s %(name)18s: %(message)s',
        stream=sys.stderr,
    )

    log = logging.getLogger('__main__')
    log.info('starting')

    # Create a limited thread pool.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    event_loop = asyncio.get_event_loop()
    try:
        event_loop.run_until_complete(run_blocking_tasks(executor))
    finally:
        event_loop.close()

    log.info('done')

