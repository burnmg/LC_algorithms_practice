# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def helper(node, parent_sum):
            if not node: return 0
            if not node.left and not node.right:
                return parent_sum * 10 + node.val
            
            res = 0 
            if node.left:
                res += helper(node.left, parent_sum*10 + node.val)
            if node.right:
                res += helper(node.right, parent_sum*10 + node.val)
            return res


        return helper(root, 0)