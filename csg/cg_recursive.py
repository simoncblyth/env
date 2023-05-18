from typing import Optional, List
from enum import Enum


class CSGOp(Enum):
    UNION = 0
    INTERSECT = 1
    DIFFERENCE = 2


class CSGNode:
    def __init__(self, left: Optional['CSGNode'], right: Optional['CSGNode'], op: CSGOp, primitive=None):
        self.left = left
        self.right = right
        self.op = op
        self.primitive = primitive

    def __repr__(self):
        if self.primitive:
            return f'CSGNode(primitive={self.primitive})'
        else:
            return f'CSGNode(left={self.left}, right={self.right}, op={self.op})'


def ray_intersect_csg_tree(node: CSGNode, p: Vector3, d: Vector3) -> Optional[float]:
    """
    Calculate the distance to the closest intersection between a ray and a CSG tree.

    :param node: The root node of the CSG tree.
    :param p: The ray origin.
    :param d: The ray direction.
    :return: The distance to the closest intersection, or None if no intersection exists.
    """
    if node.primitive is not None:
        return node.primitive.ray_intersect(p, d)

    if node.op == CSGOp.UNION:
        return min(ray_intersect_csg_tree(node.left, p, d), ray_intersect_csg_tree(node.right, p, d))
    elif node.op == CSGOp.INTERSECT:
        return max(ray_intersect_csg_tree(node.left, p, d), ray_intersect_csg_tree(node.right, p, d))
    elif node.op == CSGOp.DIFFERENCE:
        left_int = ray_intersect_csg_tree(node.left, p, d)
        if left_int is None:
            return None
        right_int = ray_intersect_csg_tree(node.right, p, d)
        if right_int is not None and right_int < left_int:
            return None
        return left_int


class Sphere:
    def __init__(self, center: Vector3, radius: float):
        self.center = center
        self.radius = radius

    def __repr__(self):
        return f'Sphere(center={self.center}, radius={self.radius})'

    def ray_intersect(self, p: Vector3, d: Vector3) -> Optional[float]:
        """
        Calculate the distance to the closest intersection between a ray and a sphere.

        :param p: The ray origin.
        :param d: The ray direction.
        :return: The distance to the closest intersection, or None if no intersection exists.
        """
        op = self.center - p
        b = op.dot(d)
        det = b ** 2 - op.dot(op) + self.radius ** 2

        if det < 0:
            return None
        else:
            det = math.sqrt(det)
            t1 = b - det
            t2 = b + det

            if t2 < 0:
                return None
            elif t1 > 0:
                return t1
            else:
                return t2

