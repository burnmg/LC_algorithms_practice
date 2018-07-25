# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        cur = root
        while cur:
            if cur.left:
                temp = cur.right
                cur.right = cur.left
                cur.left = None
                cur2 = cur.right
                while cur2.right:
                    cur2 = cur2.right
                cur2.right = temp
            cur = cur.right
            
        
    
