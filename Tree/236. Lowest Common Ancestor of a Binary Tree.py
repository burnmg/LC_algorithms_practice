# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        return self.helper(root, p, q)[1]
    
    def helper(self, root, p, q):
        
        # stop case TODO
        if not root:
            return 0, None
        
        # return (found_count, res_node)
        
        
        left_found_count, left_res_node = self.helper(root.left, p, q)
        
        
        if left_found_count == 2:
            return 2, left_res_node
                
        if left_found_count == 1 and (root.val == q.val or root.val == p.val):
            return 2, root
        
        right_found_count, right_res_node = self.helper(root.right, p, q)
        
        if right_found_count == 1 and (root.val == q.val or root.val == p.val):
            return 2, root
        
        if right_found_count == 2:
            return 2, right_res_node

        if left_found_count == 1 and right_found_count == 1:
            return 2, root

        return max(left_found_count, right_found_count, root.val == q.val or root.val == p.val), None

        
            
        
        
        