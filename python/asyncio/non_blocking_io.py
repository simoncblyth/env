#!/usr/bin/env python
"""
https://stackoverflow.com/questions/27676954/non-blocking-i-o-with-asyncio
"""
import asyncio

gethost = lambda:os.environ.get("TCP_HOST", "127.0.0.1" ) 
getport = lambda:int(os.environ.get("TCP_PORT", "15006" )) 


async def read(reader, callback):
    while True:
        data = await reader.read(2**12)
        if not data: # EOF
            break
        callback(data)
    pass


async def echo_client():
    reader, writer = await asyncio.open_connection(gethost(), getport())
    chunks = []
    asyncio.async(read(reader, chunks.append))
    count = 0 
    while True:
        count += 1 
        if chunks: 
            print(len(chunks))
        pass
        await asyncio.sleep(0.016) # advance asyncio loop
    pass

