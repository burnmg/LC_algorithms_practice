# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.helper(root)[1]
    
    def helper(self, root):
        
        if not root:
            return 0, 0, 0
        
        left_tilt, left_tilt_sum, left_sum = self.helper(root.left)
        right_tilt, right_tilt_sum, right_sum = self.helper(root.right)
        
        tilt = abs(left_sum-right_sum)
        return tilt, left_tilt_sum + right_tilt_sum + tilt, left_sum + right_sum + root.val
        