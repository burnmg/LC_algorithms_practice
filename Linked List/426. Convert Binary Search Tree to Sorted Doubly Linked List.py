"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution(object):
    def treeToDoublyList(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None
        
        _min, _max = self.recur(root)
        
        _min.left = _max
        _max.right = _min
        
        return _min
        
        
    def recur(self, root):
        
        if not root:
            return (None, None)
        
        left_min, left_max = self.recur(root.left)
        right_min, right_max = self.recur(root.right)
        _min = _max = root
        
        if left_max:
            root.left = left_max
            left_max.right = root
            _min = left_min

        if right_min:
            root.right = right_min
            right_min.left = root
            _max = right_max
        
        
        return (_min, _max)
        