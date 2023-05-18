#!/usr/bin/env python
"""

In this version, we use a stack to keep track of the nodes we need to process,
along with the range of distances (t_enter and t_exit) that we need to consider
for each node. We start with the root node and an empty stack, and iterate
until the stack is empty. For each node, we check if it is a primitive, in
which case we calculate its intersection with the ray and update t_min if
necessary. If the node is not a primitive, we handle it depending on its
operation (UNION, INTERSECT, or DIFFERENCE) and push its children onto the
stack with appropriate t_enter and t_exit values.

"""
import numpy as np
from typing import Optional, List
from typing import Tuple
from enum import Enum

from typing import Tuple


class CSGOp(Enum):
    UNION = 0
    INTERSECT = 1
    DIFFERENCE = 2



class CSGNode:
    def __init__(self, op, left=None, right=None, primitive=None):
        self.op = op
        self.left = left
        self.right = right
        self.primitive = primitive

    def intersects(self, ray_origin, ray_direction, t_min):
        if self.primitive is not None:
            return self.primitive.intersects(ray_origin, ray_direction, t_min)

        left_dist = self.left.intersects(ray_origin, ray_direction, t_min)
        right_dist = self.right.intersects(ray_origin, ray_direction, t_min )

        if left_dist <= t_min and right_dist <= t_min:
            return t_min

        if left_dist <= t_min and right_dist > t_min:
            return right_dist

        if right_dist <= t_min and left_dist > t_min:
            return left_dist

        if self.op == CSGOp.UNION:
            return min(left_dist, right_dist)

        if self.op == CSGOp.INTERSECT:
            return max(left_dist, right_dist)

        if self.op == CSGOp.DIFFERENCE:
            if left_dist < right_dist:
                return left_dist
            else:
                return None



class Sphere(object):
    def __init__(self, center, radius ):
        self.center = np.asarray( center )
        self.radius = radius

    def intersects(self, ray_origin, ray_direction, t_min=0.  ):
        O = ray_origin - self.center
        D = ray_direction
        rr = self.radius*self.radius

        b = np.dot( O, D) 
        c = np.dot( O, O) - rr
        d = np.dot( D, D) 
 
        disc = b*b-d*c
        sdisc = np.sqrt(disc) if disc > 0. else 0.   

        t0 = (-b - sdisc)/d
        t1 = (-b + sdisc)/d

        t_cand = ( t0 if t0 > t_min else t1 ) if sdisc > 0. else t_min 
        return t_cand

class Cube(object):
    def __init__(self, min_point, max_point):
        self.min_point = np.asarray(min_point)
        self.max_point = np.asarray(max_point)

    def intersects(self, ray_origin, ray_direction):
        inv_direction = 1./ ray_direction 
        tmin = (self.min_point - ray_origin) * inv_direction
        tmax = (self.max_point - ray_origin) * inv_direction
        tmin = max(tmin.x, max(tmin.y, tmin.z))
        tmax = min(tmax.x, min(tmax.y, tmax.z))
        if tmin > tmax:
            return None
        return tmin




def ray_intersect_csg_tree(node, p, d):
    """
    Calculate the distance to the closest intersection between a ray and a CSG tree.

    :param node: The root node of the CSG tree.
    :param p: The ray origin.
    :param d: The ray direction.
    :return: The distance to the closest intersection, or None if no intersection exists.
    """
    stack = [(node, None, None)]
    t_min = None

    while len(stack) > 0:
        node, t_enter, t_exit = stack.pop()

        if node.primitive is not None:
            t = node.primitive.ray_intersect(p, d)
            if t is not None:
                if t_min is None or t < t_min:
                    t_min = t
                if t_exit is not None and t_min < t_exit:
                    return t_min
            else:
                if t_enter is not None and t_exit is not None and t_exit < t_enter:
                    return None
        else:
            if node.op == CSGOp.UNION:
                stack.append((node.left, t_enter, t_exit))
                stack.append((node.right, t_enter, t_exit))
            elif node.op == CSGOp.INTERSECT:
                if t_enter is None:
                    t_enter = -float('inf')
                if t_exit is None:
                    t_exit = float('inf')

                t_left = ray_intersect_csg_tree(node.left, p, d)
                if t_left is not None and t_left > t_enter and t_left < t_exit:
                    stack.append((node.left, t_enter, t_left))
                    t_exit = min(t_exit, t_left)

                t_right = ray_intersect_csg_tree(node.right, p, d)
                if t_right is not None and t_right > t_enter and t_right < t_exit:
                    stack.append((node.right, t_enter, t_right))
                    t_exit = min(t_exit, t_right)
            elif node.op == CSGOp.DIFFERENCE:
                stack.append((node.left, t_enter, t_exit))
                t_right = ray_intersect_csg_tree(node.right, p, d)
                if t_right is not None:
                    if t_right > t_enter and (t_min is None or t_right < t_min):
                        t_min = t_right
                    if t_exit is not None and t_min < t_exit:
                        return t_min

    return t_min


def test_csg():
    sphere = Sphere([0., 0., 0.], 1.)
    cube = Cube([-1., -1., -1.], [1., 1., 1.])

    sphere_node = CSGNode(op=None,primitive=sphere)
    cube_node = CSGNode(op=None,primitive=cube)
    root_node = CSGNode(op=CSGOp.INTERSECT, left=sphere_node, right=cube_node)

    rays = [
        (Vector3(0, 0, 10), Vector3(0, 0, -1)),  # Ray points away from object
        (Vector3(0, 0, 10), Vector3(0, 0, 1)),   # Ray misses object
        (Vector3(0, 0, 10), Vector3(0, 0, -2)),  # Ray intersects object
        (Vector3(0, 0, -10), Vector3(0, 0, 1)),  # Ray intersects object from behind
    ]

    expected_results = [None, None, 9.0, None]

    for ray, expected_result in zip(rays, expected_results):
        result = ray_intersect_csg_tree(root_node, ray[0], ray[1])
        assert result == expected_result, f"Expected {expected_result}, got {result} for ray {ray}"


if __name__ == '__main__':
    test_csg()
pass

