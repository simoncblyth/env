#!/usr/bin/env python
"""
Usage::

   chroma-   # for virtualenv python setup
   ~/e/pygame/tests/pg.py


"""
import sys, pygame
from pygame.locals import *

class Dummy(object):
    def __init__(self, width=320, height=240):
        self.clicked = False
        self.done = False
        self.width = width
        self.height = height

    def __call__(self):
        pygame.init()
        size = self.width, self.height
        screen = pygame.display.set_mode(size)
        while not self.done:
            for event in pygame.event.get():
                print event
                self.process_event(event)
      
            black = 0, 0, 0
            screen.fill(black)
            #screen.blit(ball, ballrect)
            pygame.display.flip()


    def translate(self, v):
        print 'translate ',v

    def process_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 4:
                print 'MOUSEBUTTONDOWN 4'

            elif event.button == 5:
                print 'MOUSEBUTTONDOWN 5'

            elif event.button == 1:
                mouse_position = pygame.mouse.get_rel()
                print 'MOUSEBUTTONDOWN 1 CLICKED'
                self.clicked = True

        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                print 'MOUSEBUTTONUP 1 UN-CLICKED'
                self.clicked = False 

        elif event.type == MOUSEMOTION and self.clicked:
            movement = np.array(pygame.mouse.get_rel())
            print "MOUSEMOTION"

        elif event.type == KEYDOWN:
            if event.key == K_LALT or event.key == K_RALT:
                print "K_LALT or K_RALT   (option)"
            elif event.key == K_F6:
                print "K_F6"
            elif event.key == K_F7:
                print "K_F7"

            elif event.key == K_F11:
                print "K_F11      (fn+option+f11) but difficult to avoid an (option)"
                pygame.display.toggle_fullscreen()

            elif event.key == K_ESCAPE:
                print "K_ESCAPE"
                self.done = True
                return

            elif event.key == K_EQUALS:
                print "K_EQUALS"

            elif event.key == K_MINUS:
                print "K_MINUS"

            elif event.key == K_PAGEDOWN:
                print "K_PAGEDOWN   (fn+arrowdown)"

            elif event.key == K_PAGEUP:
                print "K_PAGEUP    (fn+arrowup)"

            elif event.key == K_3:
                print "K_3"
            elif event.key == K_g:
                print "K_g"

            elif event.key == K_F12:
                print "K_F12 screenshot"

            elif event.key == K_F5:
                print "K_F5"

            elif event.key == K_m:
                print "K_m movie"




if __name__ == '__main__':

    dummy=Dummy()
    dummy()



