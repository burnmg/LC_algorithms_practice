# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        _, max_path_sum = self.helper(root)
        
        return max_path_sum
    
    def helper(self, root):
        
        if not root:
            return float('-inf'), float('-inf')
        
        max_path_sum_with_left_node, left_max_path_sum = self.helper(root.left)
        max_path_sum_with_right_node, right_max_path_sum = self.helper(root.right)
        
        max_path_sum_with_root = max(root.val + max_path_sum_with_left_node, root.val + max_path_sum_with_right_node, root.val)
        max_path_sum = max([max_path_sum_with_root, root.val + max_path_sum_with_left_node + max_path_sum_with_right_node, left_max_path_sum, right_max_path_sum])
        
        return max_path_sum_with_root, max_path_sum