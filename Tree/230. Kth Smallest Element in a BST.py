# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        _, res = self.helper(root, k)
        
        return res.val
        
        
    def helper(self, root, k):
        
        if not root:
            return 0, None
        
        count_left, kthSmallest_node = self.helper(root.left, k)
        
        if kthSmallest_node:
            return -1, kthSmallest_node
        
        if count_left == k - 1:
            return -1, root
        
        count_right, kthSmallest_node = self.helper(root.right, k - count_left - 1)
        if kthSmallest_node:
            return -1, kthSmallest_node
        
        return count_left + count_right + 1, None