# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        graph = collections.defaultdict(list)
        def connect(parent, child):
            if parent and child:
                graph[parent.val].append(child.val)
                graph[child.val].append(parent.val)
            if child.left:
                connect(child, child.left)
            if child.right:
                connect(child, child.right)
            
        connect(None, root)
        level = [target.val]
        visited = set(level)
        for i in range(K):
            level = [y for x in level for y in graph[x] if y not in visited]
            visited |= set(level) 
        
        return level
        
        