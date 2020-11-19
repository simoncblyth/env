#!/usr/bin/env python
"""

https://docs.python.org/3/library/asyncio-task.html

"""
import logging
log = logging.getLogger(__name__)
import asyncio

async def nested():
    log.info("nested") 
    return 42

async def main():
    # Schedule nested() to run soon concurrently
    # with "main()".
    log.info("main") 
    task = asyncio.create_task(nested())

    # "task" can now be used to cancel "nested()", or
    # can simply be awaited to wait until it is complete:
    return await task


logging.basicConfig(level=logging.INFO)
out = asyncio.run(main())
print(out)
assert out == 42
