# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        def rec(t):
            if not t: return ''
            
            if t.left and t.right:
                return str(t.val) + '(' + rec(t.left) + ')' + '(' + rec(t.right) + ')'
            elif not t.left and t.right:
                return str(t.val) + '()' + '(' + rec(t.right) + ')'
            elif t.left and not t.right:
                return str(t.val) + '(' + rec(t.left) + ')'
            else:
                return str(t.val)
        
        return rec(t)
