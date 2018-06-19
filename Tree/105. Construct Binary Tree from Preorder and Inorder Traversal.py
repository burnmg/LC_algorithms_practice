# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        preorder = collections.deque(preorder)
        
        return self.helper(preorder, inorder)
    
    def helper(self, preorder, inorder):
        
        if not preorder or not inorder:
            return None
        
        root = TreeNode(preorder.popleft())
        i = inorder.index(root.val)
        
        root.left = self.helper(preorder, inorder[:i])
        root.right = self.helper(preorder, inorder[i+1:])
        
        return root