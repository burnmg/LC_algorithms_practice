# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth2(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if left == 0:
            return 1 + right
        
        if right == 0:
            return 1 + left
        
        return 1 + min(left, right)  
    
    def minDepth(self, root):
        
        if not root:
            return 0
        queue = collections.deque()
        queue.append((1,root))
        
        while queue:
            x = queue.popleft()
            if not x[1].left and not x[1].right:
                return x[0]
            else:
                if x[1].left:
                    queue.append((x[0]+1, x[1].left))
                if x[1].right:
                    queue.append((x[0]+1, x[1].right))
        return 0