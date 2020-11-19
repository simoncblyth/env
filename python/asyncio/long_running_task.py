#!/usr/bin/env python
"""

https://stackoverflow.com/questions/45226289/how-to-poll-python-asyncio-task-status

"""
import asyncio


async def send_heartbeat(heartbeat_interval):
    await asyncio.sleep(1) 

async def long_running_task(*args, **kwargs):
    await asyncio.sleep(100000) 

async def some_job(*args, **kwargs):
    future = asyncio.ensure_future(long_running_task(*args, **kwargs))
    while not future.done():
        await send_heartbeat(heartbeat_interval=15)
    pass
    try:
        result = future.result()
    except asyncio.CancelledError:
        print("he task has been cancelled")
    except Exception:
        print("some exception was raised in long running task")



loop = asyncio.get_event_loop()
loop.run_until_complete(some_job())
loop.close()



