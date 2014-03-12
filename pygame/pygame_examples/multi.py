"""
http://stackoverflow.com/questions/8106002/using-the-python-multiprocessing-module-for-io-with-pygame-on-mac-os-10-7



::

    delta:tests blyth$ python multi.py 
    The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
    Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.
    The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
    Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.
    The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
    Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.
    The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().



::

    Crashed Thread:  0  Dispatch queue: com.apple.main-thread

    Exception Type:  EXC_BREAKPOINT (SIGTRAP)
    Exception Codes: 0x0000000000000002, 0x0000000000000000

    Application Specific Information:
    *** multi-threaded process forked ***
    crashed on child side of fork pre-exec

    Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
    0   com.apple.CoreFoundation        0x00007fff89ec23aa __CFRunLoopServiceMachPort + 330
    1   com.apple.CoreFoundation        0x00007fff89ec1939 __CFRunLoopRun + 1161
    2   com.apple.CoreFoundation        0x00007fff89ec1275 CFRunLoopRunSpecific + 309
    3   com.apple.HIToolbox             0x00007fff8d106f0d RunCurrentEventLoopInMode + 226
    4   com.apple.HIToolbox             0x00007fff8d106b85 ReceiveNextEventCommon + 173
    5   com.apple.HIToolbox             0x00007fff8d106abc _BlockUntilNextEventMatchingListInModeWithFilter + 65
    6   com.apple.AppKit                0x00007fff8a3ca28e _DPSNextEvent + 1434
    7   com.apple.AppKit                0x00007fff8a3c98db -[NSApplication nextEventMatchingMask:untilDate:inMode:dequeue:] + 122
    8   libSDL-1.2.0.dylib              0x0000000103d2c049 QZ_PumpEvents + 372
    9   libSDL-1.2.0.dylib              0x0000000103d0cfb1 SDL_PumpEvents + 35
    10  event.so                        0x0000000103f78577 pygame_pump + 23
    11  org.python.python               0x00000001039a9fa6 PyEval_EvalFrameEx + 11702
    12  org.python.python               0x00000001039a7076 PyEval_EvalCodeEx + 1734
    13  org.python.python               0x000000010393a0c6 function_call + 342
    14  org.python.python               0x0000000103916665 PyObject_Call + 101
    15  org.python.python               0x00000001039ab4e6 PyEval_EvalFrameEx + 17142



"""
import pygame
import multiprocessing
pygame.init()

def f():
    while True:
        pygame.event.pump() #if this is replaced by pass, this code works

p = multiprocessing.Process(target=f)
p.start()

while True:
    pass
