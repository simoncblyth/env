MeshLab Distribution
======================

Tried macdeployqt to collect together all the frameworks into the .app BUT find issues.


MeshLab using wrong plugins dir
---------------------------------

After distribution testing, the MeshLab app is identifying a different plugins dir, 
resulting in failure to load COLLADA::

    simon:Contents blyth$ pwd
    /usr/local/env/graphics/meshlab/meshlab/src/distrib/meshlab.app/Contents
    simon:Contents blyth$ mv PlugIns PlugIns.disable

Renaming the wrong plugins dir allows it to find the intended ``/usr/local/env/graphics/meshlab/meshlab/src/distrib/plugins``


MeshLab Bus Error
-------------------

But get bus error, when trying to ``Render > Show Vertex Label``::

    Interval Since Last Report:          409338 sec
    Crashes Since Last Report:           8
    Per-App Interval Since Last Report:  0 sec
    Per-App Crashes Since Last Report:   1

    Date/Time:       2013-12-03 18:17:44.217 +0800
    OS Version:      Mac OS X 10.5.8 (9L31a)
    Report Version:  6
    Anonymous UUID:  0AEE87B7-11A3-4A84-B851-87CA48233147

    Exception Type:  EXC_BAD_ACCESS (SIGBUS)
    Exception Codes: KERN_PROTECTION_FAILURE at 0x0000000000000004
    Crashed Thread:  0

    Thread 0 Crashed:
    0   meshlab                         0x000233a0 MainWindow::applyDecorateMode() + 64
    1   meshlab                         0x000ef174 MainWindow::qt_static_metacall(QObject*, QMetaObject::Call, int, void**) + 2580
    2   QtCore                          0x0061ebf8 QMetaObject::activate(QObject*, QMetaObject const*, int, void**) + 2008
    3   QtGui                           0x012b771c QAction::triggered(bool) + 60
    4   QtGui                           0x012b8cf8 QAction::activate(QAction::ActionEvent) + 168
    5   QtGui                           0x01292504 QMenuBar::macUpdateMenuBar() + 404
    6   QtGui                           0x01292a1c QMenuBar::macUpdateMenuBar() + 1708
    7   com.apple.HIToolbox             0x95243c48 DispatchEventToHandlers(EventTargetRec*, OpaqueEventRef*, HandlerCallRec*) + 1484
    8   com.apple.HIToolbox             0x95242de0 SendEventToEventTargetInternal(OpaqueEventRef*, OpaqueEventTargetRef*, HandlerCallRec*) + 464
    9   com.apple.HIToolbox             0x9525fd04 SendEventToEventTarget + 52
    10  com.apple.HIToolbox             0x95295494 SendHICommandEvent(unsigned long, HICommand const*, unsigned long, unsigned long, unsigned char, OpaqueEventTargetRef*, OpaqueEventTargetRef*, OpaqueEventRef**) + 452
    11  com.apple.HIToolbox             0x952bbde0 SendMenuItemSelectedEvent + 136
    12  com.apple.HIToolbox             0x952bbcd4 FinishMenuSelection(MenuData*, MenuData*, MenuResult*, MenuResult*, unsigned long, unsigned long, unsigned long, unsigned char) + 136
    13  com.apple.HIToolbox             0x952989a8 MenuSelectCore(MenuData*, Point, double, unsigned long, OpaqueMenuRef**, unsigned short*) + 568
    14  com.apple.HIToolbox             0x95327bf4 MenuSelect + 100
    15  QtGui                           0x0124c3a8 QApplicationPrivate::globalEventProcessor(OpaqueEventHandlerCallRef*, OpaqueEventRef*, void*) + 5368
    16  com.apple.HIToolbox             0x95243c48 DispatchEventToHandlers(EventTargetRec*, OpaqueEventRef*, HandlerCallRec*) + 1484
    17  com.apple.HIToolbox             0x95242de0 SendEventToEventTargetInternal(OpaqueEventRef*, OpaqueEventTargetRef*, HandlerCallRec*) + 464
    18  com.apple.HIToolbox             0x9525fd04 SendEventToEventTarget + 52
    19  com.apple.HIToolbox             0x952733fc ToolboxEventDispatcherHandler(OpaqueEventHandlerCallRef*, OpaqueEventRef*, void*) + 2472
    20  com.apple.HIToolbox             0x9524409c DispatchEventToHandlers(EventTargetRec*, OpaqueEventRef*, HandlerCallRec*) + 2592
    21  com.apple.HIToolbox             0x95242de0 SendEventToEventTargetInternal(OpaqueEventRef*, OpaqueEventTargetRef*, HandlerCallRec*) + 464
    22  com.apple.HIToolbox             0x9525fd04 SendEventToEventTarget + 52
    23  QtGui                           0x01261aa4 QDesktopWidget::resizeEvent(QResizeEvent*) + 6052
    24  QtCore                          0x00600d3c QEventLoop::processEvents(QFlags<QEventLoop::ProcessEventsFlag>) + 108
    25  QtCore                          0x00601204 QEventLoop::exec(QFlags<QEventLoop::ProcessEventsFlag>) + 420
    26  QtCore                          0x006040bc QCoreApplication::exec() + 236
    27  meshlab                         0x00007c00 main + 3344
    28  meshlab                         0x000060f0 start + 64
    29  ???                             0x00000ffc 0 + 4092



Resolved with a simple .app rebuild::

    simon:~ blyth$ meshlab-cd
    simon:src blyth$ rm -rf distrib/meshlab.app/
    simon:src blyth$ make



