# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        
        if not root.left and not root.right:
            return [str(root.val)]
        
        res = []
        if root.left:
            left_paths = self.binaryTreePaths(root.left)
            for path in left_paths:
                res.append(str(root.val) + '->' + path)
        
        if root.right:
            right_paths = self.binaryTreePaths(root.right)
            for path in right_paths:
                res.append(str(root.val) + '->' + path)
        return res
            