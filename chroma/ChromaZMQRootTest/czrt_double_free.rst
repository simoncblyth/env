ZMQRoot/MyTMessage dtor issue
================================

::

    (chroma_env)delta:ChromaZMQRootTest blyth$ lldb $(czrt-bin)
    Current executable set to '/usr/local/env/chroma/ChromaZMQRootTest/bin/ChromaZMQRootTest' (x86_64).
    (lldb) r
    Process 36786 launched: '/usr/local/env/chroma/ChromaZMQRootTest/bin/ChromaZMQRootTest' (x86_64)
    ZMQRoot::ZMQRoot envvar [CHROMA_CLIENT_CONFIG] config [tcp://localhost:5555] 
    ChromaPhotonList::Print  [1]
    ZMQRoot::SendObject sent bytes: 217 
    ZMQRoot::ReceiveObject received bytes: 217 
    ZMQRoot::ReceiveObject reading TObject from the TMessage 
    ZMQRoot::ReceiveObject returning TObject 
    ChromaZMQRootTest(36786,0x7fff7ab15310) malloc: *** error for object 0x10561df08: pointer being freed was not allocated
    *** set a breakpoint in malloc_error_break to debug
    Process 36786 stopped
    * thread #1: tid = 0x278fe8, 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8c34e866:  jae    0x7fff8c34e870            ; __pthread_kill + 20
       0x7fff8c34e868:  movq   %rax, %rdi
       0x7fff8c34e86b:  jmpq   0x7fff8c34b175            ; cerror_nocancel
       0x7fff8c34e870:  ret    

    (lldb) bt
    * thread #1: tid = 0x278fe8, 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8c34e866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8d5f835c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff9320ab1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff903a607f libsystem_malloc.dylib`free + 411
        frame #4: 0x0000000100066f07 libCore.so`TBuffer::~TBuffer() + 39
        frame #5: 0x0000000100043155 libZMQRoot.dylib`MyTMessage::~MyTMessage() + 21
        frame #6: 0x0000000100043135 libZMQRoot.dylib`MyTMessage::~MyTMessage() + 21
        frame #7: 0x0000000100044eb5 libZMQRoot.dylib`ZMQRoot::ReceiveObject() + 469
        frame #8: 0x0000000100005e02 ChromaZMQRootTest`main + 370

