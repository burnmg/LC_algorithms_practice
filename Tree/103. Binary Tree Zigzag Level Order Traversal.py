# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        queue = [(0,root)]
        result = []
        results = []
        current_depth = 0
        to_right = True
        
        while queue:
            x = queue.pop(0)
            
            if current_depth != x[0]:
                current_depth = x[0]
                to_right = not to_right
                results.append(result)
                result = []
            
            if to_right:
                result.append(x[1].val)
            else:
                result.insert(0, x[1].val)

            if x[1].left:
                queue.append((x[0]+1, x[1].left))
            if x[1].right:
                queue.append((x[0]+1, x[1].right))       
        
        results.append(result)
        
        return results
                