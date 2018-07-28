# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
        return self.get_leaves_values(root1) == self.get_leaves_values(root2)
        
    def get_leaves_values(self, root):
        
        if not root:
            return []
        
        if not root.left and not root.right:
            return [root.val]
        
        return self.get_leaves_values(root.left) + self.get_leaves_values(root.right) 
        
            
        