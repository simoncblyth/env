#!/usr/bin/env python
"""
  http://bytes.com/topic/python/answers/23279-using-mvc-when-model-dynamic 
  wiki:AberdeenOnlineEventDisplay 
 
  Demo a basic MVC app structure in which
  The model is a simulation running in its own thread.
  The model undergoes frequent, View-able state changes

"""

import sys, sets, threading, select, math


class Observable(object):
    """An Observable notifies its observers whenever its value changes.
    Observers are just Python callables having the signature
    callMe(sender)

    Lots of MT-safe overhead, here. So...
    TO DO: Demonstrate asynchronous, coalesced notifications."""
    def __init__(self):
        self._observers = sets.Set()
        self._lock = threading.RLock()
        self._value = None

    def addObserver(self, newObserver):
        self._observers.add(newObserver) # This oughtta be locked...

    def removeObserver(self, anObserver):
        self._observers.remove(anObserver) # This oughtta be locked...

    def _notify(self):
        for observer in self._observers: # This oughtta be locked...
            try:
                observer(self)
            except:
                pass # Don't let one broken observer gum up everything

    def _getValue(self):
        self._lock.acquire()
        result = self._value
        self._lock.release()
        return result

    def _setValue(self, newValue):
        self._lock.acquire()
        self._value = newValue
        self._lock.release()
        self._notify()

    value = property(_getValue, _setValue, None, "The observable value")





class DelayedObservable(Observable):
    def __init__(self, notifyInterval=1.0):
        self._tLastChange = time.time()
        self._notifyInterval = notifyInterval # Seconds

    def _notify(self, force=0):
        dt = time.time() - self._tLastChange
        if force or (dt >= self._notifyInterval):
            for observer in self._observers:
                try:
                     observer(self)
                except:
                    pass            
        self._tLastChange = time.time()



class Model(threading.Thread):
    """Computes new values asynchronously. Notifies observers whenever
        its state changes."""
    def __init__(self, **kw):
        threading.Thread.__init__(self, **kw)
        self._stopped = 0
        self._state = Observable()

    def onStateChange(self, observer):
        self._state.addObserver(observer)

    def removeStateChange(self, observer):
        self._state.removeObserver(observer)

    def run(self):
        """Run the model in its own thread."""
        self._stopped = 0
        i = 0.0
        di = math.pi / 8.0
        while not self._stopped:
            self._state.value = math.sin(i)
            i += di

    def stop(self):
        self._stopped = 1


class View:
    """Dummy 'view' just prints the model's current value whenever
       that value changes, and responds to keyboard input."""
    def __init__(self):
        self._onQuitCB = None

    def modelStateChanged(self, modelState):
        valueBar = " " * int((1 + modelState.value) * 10)
        print "%s#" % valueBar

    def onQuit(self, newOnQuitCB):
        self._onQuitCB = newOnQuitCB

    def handleInput(self, userInput):
        if userInput.lower().startswith("q"):
            if self._onQuitCB:
                self._onQuitCB(self)


class App:
    """This sample application computes and displays garbage, at
       a high rate of speed, until the user quits."""
    def __init__(self):
        # Yep, this is really a controller and not just an app runner.
        self._model = Model()
        self._view = View()
        self._terminated = 0

        self._model.onStateChange(self._view.modelStateChanged)
        self._view.onQuit(self._quitApp)

    def run(self):
        self._model.start()
        self._terminated = 0
        while not self._terminated:
            ins, outs, errs = select.select([sys.stdin], [], [])
            if ins:
                self._view.handleInput(raw_input())
        self._model.join()

    def _quitApp(self, *args):
        self._terminated = 1
        self._model.stop()
        self._model.removeStateChange(self._view.modelStateChanged)


def main():
    """Module mainline (for standalone execution)"""
    theApp = App()
    theApp.run()

if __name__ == "__main__":
    main()
