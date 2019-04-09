# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        def dp(node):
            """
            return have_camera_no_need, no_camera_no_need, no_camera_need
            """
            
            if not node:
                return float("inf"), 0, float("inf")
            
            if not node.left and not node.right:
                return 1, float("inf"), 0
            
            left_res = dp(node.left)
            right_res = dp(node.right)
            
            have_camera_no_need = 1 + min(left_res) + min(right_res)
            no_camera_no_need = min(left_res[0] + min(right_res[:2]), right_res[0] + min(left_res[:2]))
            no_camera_need = left_res[1] + right_res[1]
            
            return have_camera_no_need, no_camera_no_need, no_camera_need
        
        return min(dp(root)[:2])
