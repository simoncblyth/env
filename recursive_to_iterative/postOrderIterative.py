#!/usr/bin/env python
"""

Hmm descoping to support complete binary trees up to maximum depth
of three/four would certainly cover all reasonable single 
solid boolean combinations.
Assuming implement transforms in a way that doesnt enlarge the tree.

* http://www.geeksforgeeks.org/iterative-postorder-traversal/





::

      1
 
      2             3

      4      5      6      7          =>   4 5 2 6 7 3 1

      8  9   10 11  12 13  14 15      =>   (8 9 4) (10 11 5) 2 (12 13 6) (14 15 7) 3 1
                                               (4         5  2)      (6         7  3)
                                                            (2                     3  1)  


csg ray trace algo 

* knowing tree depth can jumpt straight to primitives
* visit left/right/parent where left/right are primitives 

intersect ordering 

* (8   9 4)   pushLeft -> "4"   
* (10 11 5)   pushRight -> "5"
* (4  5  2)   popLeft/popRight -> pushLeft "2"    

    ( "4" and "5" child intersects no longer needed, after parent "2" intersect is computed)

* (12 13 6)    pushLeft -> "6" 
* (14 15 7)    pushRight -> "7"
* (6 7 3)      popLeft,popRight -> pushRight "3" 


* (2 3 1)     popLeft, popRight -> "1"



::

     1

     2                                            3
 
     4             5                              6           7

     8       9     10     11                     12    13     14     15
 
     16 17  18 19  20 21  22 23                  24 25 26 27  28 29  30 31 


            postorder ->  (16 17 8) (18 19 9) (4)     (20 21 10) (22 23 11) [5] [2]       24 25 12 26 27 13 6 28 29 14 30 31 15 7 3 1
                                 L8        R9  L4           L4,10       R11 R5 L2

               (16 17  8)->pushL-8 
               (18 19  9)->pushR-9  
               popL-8,popR-9->pushL[4]     L{4}   

               (20 21 10)->pushL-10 
               (22 23 11)->pushR-11 
               popL-10,popR-11->pushR[5]   R{5}

                popL-4,popR-5 (4, 5, 2) -> pushL [2]    L{2}     when left and right are > 0, time to pop-em ? 
                        

               (24 25 12)->pushL-12
               (26 27 13)->pushR-13  
                      [6]  popL-12,popR-13 -> pushL [6]      L{2,6}
               (28 29 14)->pushL-14                          L{2,6,14} 
               (30 31 15)->pushR-15                          R{15}
                      [7]  popL-14, popR-15 -> pushR [7]     L{2,6} R{7} 
                      [3]  popL-6,  popR-7  -> pushR [3]     L{2} R{3}

                    popL-2, popR-3  ->  1
                       1


It looks like using L and R intersect stacks will allow to iteratively 
evaluate the ray intersect with the binary tree just by following 
the postorder traverse while pushing and popping from the L/R stacks
which need to be able to hold a maximum of 3 entries.




With 1-based node index, excluding root at node 1 
* left always even
* right always odd


 
*   (16 17 8)  lstack:[8]
*   (18 19 9)  rstack:[9]
*   ( 8  9 



"""

 
class Node(object):
    def __init__(self, d, l=None, r=None):
        self.d = d
        self.l = l
        self.r = r

    def __repr__(self):
        if self.l is not None and self.r is not None:
            return "Node(%d,l=%r,r=%r)" % (self.d, self.l, self.r)
        elif self.l is None and self.r is None:
            return "Node(%d)" % (self.d)
        else:
            assert 0


 
def postOrderIterative(root): 
    """
    # iterative postorder traversal using
    # two stacks : nodes processed 
    """ 

    if root is None:
        return         
     
    nodes = []
    s = []
     
    nodes.append(root)
     
    while len(nodes) > 0:
         
        node = nodes.pop()
        s.append(node)
     
        if node.l is not None:
            nodes.append(node.l)
        if node.r is not None :
            nodes.append(node.r)
 
    #while len(s2) > 0:
    #    node = s2.pop()
    #    print node.d,
 
    return list(reversed(s))



root2 = Node(1, 
                l=Node(2, 
                          l=Node(4), 
                          r=Node(5)
                      ), 
                r=Node(3, l=Node(6), 
                          r=Node(7)
                      ) 
            )



root3 = Node(1, 
                l=Node(2, 
                          l=Node(4,
                                    l=Node(8),
                                    r=Node(9)
                                ), 
                          r=Node(5,
                                    l=Node(10),
                                    r=Node(11),
                                )
                      ), 
                r=Node(3, l=Node(6,
                                    l=Node(12),
                                    r=Node(13),
                                ), 
                          r=Node(7,
                                    l=Node(14),
                                    r=Node(15)
                                )
                      ) 
            )


root4 = Node(1, 
                l=Node(2, 
                          l=Node(4,
                                    l=Node(8, 
                                              l=Node(16),
                                              r=Node(17)
                                          ),
                                    r=Node(9,
                                              l=Node(18),
                                              r=Node(19)
                                          )
                                ), 
                          r=Node(5,
                                    l=Node(10,
                                              l=Node(20),
                                              r=Node(21)
                                          ),
                                    r=Node(11,
                                              l=Node(22),
                                              r=Node(23)
                                          ),
                                )
                      ), 
                r=Node(3, l=Node(6,
                                    l=Node(12,
                                              l=Node(24),
                                              r=Node(25)
                                          ),
                                    r=Node(13,
                                              l=Node(26),
                                              r=Node(27)
                                          ),
                                ), 
                          r=Node(7,
                                    l=Node(14,
                                              l=Node(28),
                                              r=Node(29)
                                          ),
                                    r=Node(15,
                                              l=Node(30),
                                              r=Node(31)
                                          )
                                )
                      ) 
            )




def binary_calc(node, left, right):
    if left and right:
        return "[%s](%s,%s)" % ( node.d, left, right )
    else:
        return "%s" % node.d


def postordereval_r(node):
    """
    * :google:`tree calculation postorder traversal`
    * http://interactivepython.org/runestone/static/pythonds/Trees/TreeTraversals.html
    """
    if not node: return

    l = postordereval_r(node.l)
    r = postordereval_r(node.r)

    return binary_calc(node, l, r )


def portordereval_i(node):
    """
    task is to recreate the output _r with _i without using recursion 
    """
    nodes = postOrderIterative(node)

    lhs = []
    rhs = []

    nn = len(nodes)

       
        






nodes = postOrderIterative(root2)
print "root2 " + " ".join(map(lambda node:str(node.d), nodes))
print 
print postordereval_r(root2)
print postordereval_i(root2)

nodes = postOrderIterative(root3)
print "root3 " + " ".join(map(lambda node:str(node.d), nodes))
print 
print postordereval_r(root3)

nodes = postOrderIterative(root4)
print "root4 " + " ".join(map(lambda node:str(node.d), nodes))
print 
print postordereval_r(root4)


for node in nodes:
    print node



