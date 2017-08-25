#pragma once

#include "DEMO_API_EXPORT.hh"

#include "GLEQ.hh" 

struct DEMO_API Frame 
{
   enum{ NUM_KEYS = 512 } ; 
   bool keys_down[NUM_KEYS] ; 

   GLFWwindow* window ;
   Frame(const char* title, int width=1024, int height=768);
   void init(const char* title, int width, int height);

   void listen();
   void handle_event(GLEQevent& event);
   void key_pressed(unsigned key);
   void key_released(unsigned key);

   void destroy();
};



