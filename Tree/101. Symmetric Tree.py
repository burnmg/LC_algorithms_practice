# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def sym(a,b):
            if a is None and b is None:
                return True
            elif a is None or b is None:
                return False
            
            return a.val == b.val and sym(a.right, b.left) and sym(a.left, b.right)
        
        if not root:
            return True
        else:
            return sym(root.left, root.right)
        