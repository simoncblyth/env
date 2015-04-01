#include "npyQueue.hh"


npyQueue::npyQueue() {
    m_queue.clear();
}

void npyQueue::push(npyBuffer* item ) 
{
    m_queue.push_front(item);
}   

bool npyQueue::isEmpty() 
{
    return m_queue.empty();
}

npyBuffer* npyQueue::pop() 
{
    npyBuffer* temp;
    temp = m_queue.front();
    m_queue.pop_front();
    return temp;
}

