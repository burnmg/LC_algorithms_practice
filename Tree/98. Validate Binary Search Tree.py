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
        if not root:
            return True
        return self.helper(root)[2]
        
    
    def helper(self, root):
        """
        return min_val, max_val, is_valid
        """
        if not root.left and not root.right:
            return root.val, root.val, True
        
        l_min, l_max = None, None
        r_min, r_max = None, None
        is_valid = True
        if root.left:
            l_min, l_max, l_valid = self.helper(root.left)
            if l_valid == False:
                return 0, 0, False
            is_valid = is_valid and l_max < root.val
            if not is_valid:
                return 0, 0, False
        else:
            l_min = root.val
        
        if root.right:
            r_min, r_max, r_valid = self.helper(root.right)
            if r_valid == False:
                return 0, 0, False
            is_valid = is_valid and r_min > root.val
        else:
            r_max = root.val
        
        
        return l_min, r_max, is_valid