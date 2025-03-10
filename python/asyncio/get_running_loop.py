# https://docs.python.org/3/library/asyncio-task.html

import asyncio
import datetime

async def display_date():
    loop = asyncio.get_running_loop()
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(1)
    pass

for i in range(10000): print(i)
asyncio.run(display_date())
