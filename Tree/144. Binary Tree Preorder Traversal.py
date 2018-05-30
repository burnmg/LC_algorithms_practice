# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def preorderTraversal2(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)
    
    def preorderTraversal(self, root):
        
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            x = stack.pop()
            
            res.append(x.val)

            if x.right:
                stack.append(x.right)
            if x.left:
                stack.append(x.left)
        return res