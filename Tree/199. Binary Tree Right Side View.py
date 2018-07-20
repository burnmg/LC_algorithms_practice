# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        res = []
        prev_node = [0, root]
        queue = collections.deque()
        
        if root.left:
            queue.append([1,root.left])
        if root.right:
            queue.append([1,root.right])
        
        while queue:
            node = queue.popleft()
            if node[0] != prev_node[0]:
                res.append(prev_node[1].val)
            if node[1].left:
                queue.append([node[0] + 1, node[1].left])
            if node[1].right:
                queue.append([node[0] + 1, node[1].right])
            prev_node = node
            
        res.append(prev_node[1].val)
        return res 
            
            