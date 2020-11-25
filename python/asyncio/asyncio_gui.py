#!/usr/bin/env python
"""

* :google:`using python asyncio with GUI runloop`

* https://stackoverflow.com/questions/47895765/use-asyncio-and-tkinter-or-another-gui-lib-together-without-freezing-the-gui

* https://pypi.org/project/aconsole/


"""
from tkinter import *
from tkinter import messagebox
import asyncio
import threading
import random
import logging
log = logging.getLogger(__name__)

def _asyncio_thread(async_loop):
    async_loop.run_until_complete(do_urls())

def do_tasks(async_loop):
    """ Button-Event-Handler starting the asyncio part. """
    threading.Thread(target=_asyncio_thread, args=(async_loop,)).start()

async def one_url(url):
    """ One task. """
    sec = random.randint(1, 8)
    await asyncio.sleep(sec)
    return 'url: {}\tsec: {}'.format(url, sec)

async def do_urls():
    """ Creating and starting 10 tasks. """
    tasks = [one_url(url) for url in range(10)]
    completed, pending = await asyncio.wait(tasks)
    results = [task.result() for task in completed]
    print('\n'.join(results))

def do_freezed():
    messagebox.showinfo(message='Tkinter is reacting.')

def main(async_loop):
    root = Tk()
    Button(master=root, text='Asyncio Tasks', command= lambda:do_tasks(async_loop)).pack()
    buttonX = Button(master=root, text='Freezed???', command=do_freezed).pack()
    log.info("[root.mainloop")
    root.mainloop()
    log.info("]root.mainloop")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    async_loop = asyncio.get_event_loop()
    main(async_loop)
