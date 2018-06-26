# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        
        if not root:
            return None 
        
        if root.val == p.val or root.val == q.val:
            return root
        
        a = self.lowestCommonAncestor(root.left, p, q)
        b = self.lowestCommonAncestor(root.right, p, q)
        if a and b:
            return root
        elif a:
            return a
        else:
            return b
        
        
