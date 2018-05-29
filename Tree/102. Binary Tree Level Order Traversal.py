# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root): # Faster
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        result = []
        results = []
        
        queue = [(0, root)]
        current_depth = 0
        while queue:
            x = queue.pop(0)
            if x[0] != current_depth:
                results.append(result)
                result = []
                current_depth = x[0]
            result.append(x[1].val)

            
            if x[1].left:
                queue.append((x[0]+1, x[1].left))
            if x[1].right:
                queue.append((x[0]+1, x[1].right))
        
        if len(result) > 0:
            results.append(result)
            
        return results
            
    def levelOrder2(self, root):
        
        if not root:
            return []
        res, cur = [[root.val]], [root]
        
        while cur:
            level = []
            new_cur = []
            for node in cur:
                if node.left:
                    level.append(node.left.val)
                    new_cur.append(node.left)
                if node.right:
                    level.append(node.right.val)
                    new_cur.append(node.right)
            cur = new_cur
            if len(level) > 0:
                   res.append(level)
        return res
            