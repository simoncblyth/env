#pragma once

#include <deque>
#include <iostream>

#include "npyBuffer.hh"

class npyQueue {
public:         
    npyQueue();

    void push(npyBuffer* item);
    npyBuffer* pop();
    bool isEmpty();

private:        
    std::deque<npyBuffer*> m_queue;
};



