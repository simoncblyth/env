#!/usr/bin/env python
"""
1-sync-async-execution.py
---------------------------
# py3.7+ required

https://medium.com/@yeraydiazdiaz/asyncio-for-the-working-python-developer-5c468e6e2e8e

"""
import asyncio

async def foo():
    print('Running in foo')
    await asyncio.sleep(0)
    print('Explicit context switch to foo again')

async def bar():
    print('Explicit context to bar')
    await asyncio.sleep(0)
    print('Implicit context switch back to bar')

async def main():
    tasks = [foo(), bar()]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main())

