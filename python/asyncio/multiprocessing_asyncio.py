#!/usr/bin/env python
"""

John Reese - Thinking Outside the GIL with AsyncIO and Multiprocessing - PyCon 2018

https://www.youtube.com/watch?reload=9&v=0kXaLh8Fz3k&feature=youtu.be&t=10m30s

Talk referenced from 

https://realpython.com/async-io-python/

http://github.com/jreese/aiomultiprocess

aio


https://bit.ly/asyncpython

https://gist.github.com/miguelgrinberg/f15bc03471f610cfebeba62438435508




"""

import logging
import multiprocessing
import asyncio
import aiohttp
log = logging.getLogger(__name__)

async def fetch_url(url):
    return await aiohttp.request('GET', url)


def fetch_all(urls):
    tx, rx = Queue(), Queue()
    Process(
       target=bootstrap,
       args=(tx,rx),
    ).start()

    for url in urls:
        task = fetch_url, (url,), {}
        tx.put_nowait(task)

    ## consume result queue  

# tasks and results 
async def run_loop(tx, rx):
    log.info("real work")

    limit = 10 
    pending = set()
    while True:
       while len(pending) < limit:
           task = tx.get_nowait()
           fn, args, kwargs = task    # fn is a coro
           pending.add(fn(*args, **kwargs))
       pass
       done, pending = await asyncio.wait(pending, ... )
       for future in done:
           rx.put_nowait(await future)
       pass           

def bootstrap(tx,rx):
    log.info("[bootstrap")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_loop(tx,rx))
    log.info("]bootstrap")

def main():
    p = multiprocessing.Process(target=bootstrap, args=("hello","world"))
    log.info("[main.start")
    p.start()
    log.info("]main.start")


############ instead of above manual approah better to use Pool


class Pool:
   async def queue(self, fn, *args, **kwargs) -> int: ... 
   async def result(self, id) -> Any: ... 
   
   async def map(self, fn, items):
       task_ids = [ await self.queue(fn, (item,), {}) for item in items ]
       return [ await self.result(task_id) for task_id in task_ids ]


async def fetch_url(url):
    return await aiohttp.request('GET', url)

async def fetch_all(urls):
    async with Pool() as pool:
        resultd = await pool.map(fetch_url, urls) 



if __name__ == '__main__':
    fmt = "p %(process)d %(processName)20s t %(thread)d tn %(threadName)20s : %(message)s" 
    logging.basicConfig(level=logging.INFO, format=fmt)
    main()
    log.info("after main")


