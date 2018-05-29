# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def construct(nums, start, end):
            if start > end:
                return None
            
            mid_i = (start+end) // 2
            root = TreeNode(nums[mid_i])
            root.left = construct(nums, start, mid_i-1)
            root.right = construct(nums, mid_i+1, end)
            
            return root
        
        return construct(nums, 0, len(nums)-1)
    