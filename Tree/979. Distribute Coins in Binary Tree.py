# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distributeCoins(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        """
        
        """
        res = [0]
        def rec(root, res):
            
            if not root: return 0
            left = rec(root.left, res)
            right = rec(root.right, res)
            res[0] += abs(left) + abs(right)
            
            return root.val + left + right - 1
                
            
        rec(root, res)
        return res[0]