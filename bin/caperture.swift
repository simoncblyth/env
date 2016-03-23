#!/usr/bin/env xcrun swift
// https://raw.githubusercontent.com/mattburns/caperture/master/caperture.swift


/*

/System/Library/Frameworks/CoreGraphics.framework/Versions/A/Headers/CGEventTypes.h::

    100 /* Constants that specify the different types of input events. */
    101 enum {
    102   /* The null event. */
    103   kCGEventNull = NX_NULLEVENT,
    104 
    105   /* Mouse events. */
    106   kCGEventLeftMouseDown = NX_LMOUSEDOWN,
    107   kCGEventLeftMouseUp = NX_LMOUSEUP,
    108   kCGEventRightMouseDown = NX_RMOUSEDOWN,
    109   kCGEventRightMouseUp = NX_RMOUSEUP,
    110   kCGEventMouseMoved = NX_MOUSEMOVED,
    111   kCGEventLeftMouseDragged = NX_LMOUSEDRAGGED,
    112   kCGEventRightMouseDragged = NX_RMOUSEDRAGGED,
    113 


/System/Library/Frameworks/IOKit.framework/Versions/A/Headers/hidsystem/IOLLEvent.h::

     82 /* mouse events */
     83 
     84 #define NX_LMOUSEDOWN       1   /* left mouse-down event */
     85 #define NX_LMOUSEUP     2   /* left mouse-up event */
     86 #define NX_RMOUSEDOWN       3   /* right mouse-down event */
     87 #define NX_RMOUSEUP     4   /* right mouse-up event */
     88 #define NX_MOUSEMOVED       5   /* mouse-moved event */
     89 #define NX_LMOUSEDRAGGED    6   /* left mouse-dragged event */
     90 #define NX_RMOUSEDRAGGED    7   /* right mouse-dragged event */
     91 #define NX_MOUSEENTERED     8   /* mouse-entered event */
     92 #define NX_MOUSEEXITED      9   /* mouse-exited event */



/System/Library/Frameworks/CoreGraphics.framework/Versions/A/Headers/CGEvent.h::


     32 /* Return a new mouse event.
     33 
     34    The event source may be taken from another event, or may be NULL.
     35    `mouseType' should be one of the mouse event types. `mouseCursorPosition'
     36    should be the position of the mouse cursor in global coordinates.
     37    `mouseButton' should be the button that's changing state; `mouseButton'
     38    is ignored unless `mouseType' is one of `kCGEventOtherMouseDown',
     39    `kCGEventOtherMouseDragged', or `kCGEventOtherMouseUp'.
     40 
     41    The current implemementation of the event system supports a maximum of
     42    thirty-two buttons. Mouse button 0 is the primary button on the mouse.
     43    Mouse button 1 is the secondary mouse button (right). Mouse button 2 is
     44    the center button, and the remaining buttons are in USB device order. */
     45 
     46 CG_EXTERN CGEventRef CGEventCreateMouseEvent(CGEventSourceRef source,
     47   CGEventType mouseType, CGPoint mouseCursorPosition,
     48   CGMouseButton mouseButton) CG_AVAILABLE_STARTING(__MAC_10_4, __IPHONE_NA);


    308 /* Post an event into the event stream at a specified location.
    309 
    310    This function posts the specified event immediately before any event taps
    311    instantiated for that location, and the event passes through any such
    312    taps. */
    313 
    314 CG_EXTERN void CGEventPost(CGEventTapLocation tap, CGEventRef event)
    315   CG_AVAILABLE_STARTING(__MAC_10_4, __IPHONE_NA);
    316 



*/



import Foundation
 
// Start QuickTime Player using AppleScript
func startQT() {
    var scriptToPerform: NSAppleScript?
    let asCommand = "tell application \"QuickTime Player\" \n" +
            " activate \n" +
            " new screen recording \n" +
            " delay 1 \n" +
            " tell application \"System Events\" to key code 49 \n" +
            " delay 1\n" +
            " end tell"

    scriptToPerform = NSAppleScript(source:asCommand)
    let errorInfo = AutoreleasingUnsafeMutablePointer<NSDictionary?>()

    if let script = scriptToPerform {
        script.executeAndReturnError(errorInfo)
    }
}



// Click and drag the mouse as defined by the supplied commanline arguments
func dragMouse() {
    let args = NSUserDefaults.standardUserDefaults()

    let x  = CGFloat(args.integerForKey("x"))
    let y  = CGFloat(args.integerForKey("y"))
    let w = CGFloat(args.integerForKey("w"))
    let h = CGFloat(args.integerForKey("h"))
 
    let p0 = CGPointMake(x, y)
    let p1 = CGPointMake(x + w, y + h)


    // as do not want to upgrade now.. hardcode some enum values according to 
    // http://stackoverflow.com/questions/31943951/swift-and-my-idle-timer-implementation-missing-cgeventtype
    // and C headers
    //
    //let leftMouseDown = CGEventType.LeftMouseDown
    //let leftMouseDragged = CGEventType.LeftMouseDragged
    //let leftMouseUp = CGEventType.LeftMouseUp
    //let mouseButtonLeft = CGMouseButton.Left 
    
    //let leftMouseDown = CGEventType(rawValue: 1)!    
    //let leftMouseDragged = CGEventType(rawValue: 6)!    
    //let leftMouseUp = CGEventType(rawValue: 2)!

    let leftMouseDown = kCGEventLeftMouseDown 
    let leftMouseDragged = kCGEventLeftMouseDragged
    let leftMouseUp = kCGEventLeftMouseUp 
    let mouseButtonLeft = CGMouseButton(kCGMouseButtonLeft)

 
    let mouseDown = CGEventCreateMouseEvent(nil, leftMouseDown, p0, mouseButtonLeft).takeUnretainedValue()
    let mouseDrag = CGEventCreateMouseEvent(nil, leftMouseDragged, p1, mouseButtonLeft).takeUnretainedValue()
    let mouseUp = CGEventCreateMouseEvent(nil, leftMouseUp, p1, mouseButtonLeft).takeUnretainedValue()
 
    let kDelayUSec : useconds_t = 500_000
   

    //let hidEventTap = CGEventTapLocation.CGHIDEventTap  
    let hidEventTap = CGEventTapLocation(kCGHIDEventTap)


    CGEventPost(hidEventTap, mouseDown)
    usleep(kDelayUSec)
    CGEventPost(hidEventTap, mouseDrag)
    usleep(kDelayUSec)
    CGEventPost(hidEventTap, mouseUp)
}

 
if (Process.arguments.count != 9) {
    print("usage:")
    print("    ./caperture.swift -x 100 -y 100 -w 400 -h 300")
} else {
    startQT()
    dragMouse()
}

