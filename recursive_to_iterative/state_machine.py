#!/usr/bin/env python
"""
:google:`convert recursive to iterative python`


tail call

* https://en.wikipedia.org/wiki/Tail_call

if control must return to the caller to allow it to inspect or modify the return value before returning it
then its not a tail call.


"""

class State(object):
    start = 0 
    cont = 1
    done = 2

def fact_r(n):
    if n < 2:
        return 1
    return n * fact_r(n-1)

def fact_interpreter(n):
    """
    http://stupidpythonideas.blogspot.tw/2014/10/how-do-i-make-recursive-function.html

    Here's what a function call does: 

    * The caller pushes the "program counter" and the arguments onto the stack, then it jumps to the callee. 
    * The callee pops, computes the result, pushes the result, jumps to the popped counter. 

      The only issue is that the callee can have locals that shadow the caller's; you can
      handle that by just pushing all of your locals (not the post-transformation locals, 
      which include the stack itself, just the set used by the recursive function) as well.

    """
    state = State.start
    stack = [(State.done, n)]   ## start with [(pc, n)]
    while True:
        if state == State.start:
            pc, n = stack.pop()
            if n < 2:
                stack.append(1)          # return 1,  to return push the value and record state
                state = pc
                continue
            stack.append((pc, n))            # stash locals
            stack.append((State.cont, n-1))  # call recursively
            state = State.start
            continue
        elif state == State.cont:
            retval = stack.pop()         # get return value
            pc, n = stack.pop()          # restore locals
            stack.append(n * retval)     # return n * fact(n-1)
            state = pc
            continue
        elif state == State.done:
            retval = stack.pop()
            return retval


def fact_cleaner(n):
    """
    Simplify by using registers (local vars) rather than just the stack

    * pass arguments in registers
    * return values in registers
 
    """
    state = State.start
    pc = State.done
    stack = []
    while True:
        if state == State.start:
            if n < 2:
                retval = 1 # return 1
                state = pc
                continue
            pass
            stack.append((pc, n))
            pc, n, state = State.cont, n-1, State.start
        elif state == State.cont:
            state, n = stack.pop()
            retval = n * retval
        elif state == State.done:
            return retval


class Tree(object):
   def __init__(self, v, l=None, r=None ):
       self.v = v
       self.l = l 
       self.r = r 


def size_recur(t):
    """Returns the number of nodes in binary tree T.
    >>> t = Tree(10, Tree(4, Tree(1), Tree(7)), Tree(16, Tree(13), Tree(19)))
    >>> size_recur(t)
    7
    """
    if t is None:
        return 0
    return 1 + size_recur(t.l) + size_recur(t.r)
 
def size_iter(t):
    """
    * https://www.ocf.berkeley.edu/~shidi/cs61a/wiki/Iteration_vs._recursion

    Returns the number of nodes in binary tree T.
    >>> t = Tree(10, Tree(4, Tree(1), Tree(7)), Tree(16, Tree(13), Tree(19)))
    >>> size_iter(t)
    7

    Finding size of binary tree: simple as just needs to visit each node. 


    """
    count = 0
    nodes = [] # unprocessed nodes
    if t is not None:
        nodes.append(t)

    while nodes:        # while nodes is not empty
        u = nodes.pop() # get an unprocessed node
        count += 1      # increment counter

        if u.l:
            nodes.append(u.l) 
        if u.r:
            nodes.append(u.r)
 
    return count





if __name__ == '__main__':
    n = 5
    nr = fact_r(n) 
    ni = fact_interpreter(n)
    assert ni == nr
    nc = fact_cleaner(n)
    assert nc == nr
    print nr

    t = Tree(10, 
                Tree(4, 
                        Tree(1), 
                        Tree(7)
                    ), 
                Tree(16, 
                        Tree(13), 
                        Tree(19)
                    )
            )

    assert size_recur(t) == size_iter(t)





