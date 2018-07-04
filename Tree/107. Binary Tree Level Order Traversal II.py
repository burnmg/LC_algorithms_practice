# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue = collections.deque()
        
        queue.append([0, root])
        results = []
        result = []
        cur_level = 0
        while queue:
            pair = queue.popleft()
            
            if pair[0] != cur_level:
                results.append(result)
                cur_level = pair[0]
                result = []
                
            result.append(pair[1].val)
            
            if pair[1].left:
                queue.append([cur_level+1, pair[1].left])
                
            if pair[1].right:
                queue.append([cur_level+1, pair[1].right])
        
        results.append(result)
            
        return results[::-1]
        