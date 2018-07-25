# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        res, _, _ = self.helper(root)
        return res
        
    def helper(self, root):
        
        if not root:
            return True, None, None
        
        validate_left = True
        left_min = None 
        
        if root.left:
            left_is_val, left_min, left_max = self.helper(root.left)
            validate_left = left_is_val and root.val > left_max
        
        validate_right = True 
        right_max = None
        if root.right:
            right_is_val, right_min, right_max = self.helper(root.right)
            validate_right = right_is_val and root.val < right_min
        
        return (validate_left and validate_right), left_min if left_min else root.val, right_max if right_max else root.val
        
        
        
        