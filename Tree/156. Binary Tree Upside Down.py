# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def upsideDownBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        
        return self.helper(root)[0]
    def helper(self, root):
        
        """
        return new_root, right_most_node
        """
        
        if not root.left and not root.right:
            return root, root
        
        l_new_root, l_right_most_node = self.helper(root.left)
        
        new_root = l_new_root
        l_right_most_node.left = root.right
        l_right_most_node.right = root
        root.left = None
        root.right = None
        
        return new_root, root
        