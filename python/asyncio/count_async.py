# https://realpython.com/async-io-python/
import asyncio


async def noop():
    pass

async def count():
    for i in range(1000):
        print("count.%d" % i )
        await asyncio.sleep(0.2)
        print("count.%d" % i )

async def block():
    print("block.One")
    time.sleep(1)
    print("block.Two")

async def other():
    for i in range(1000):
        print("other.%d" % i)
        await asyncio.sleep(0.5)
        print("other.%d" % i )

async def loopy():
    for i in range(10000):
        print("loopy.%d"%i)
        await asyncio.sleep(0.001)
    pass


async def main():
    await asyncio.gather(loopy(),count(), noop(), count(), block(), count(), other())

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
