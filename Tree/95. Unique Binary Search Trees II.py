# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        def generate(values):
            if len(values) == 1:
                return [TreeNode(values[0])]
            if len(values) == 0:
                return []
            
            res = []
            for i, x in enumerate(values):
                left_set = generate(values[:i])
                right_set = generate(values[i+1:])
                
                if len(left_set) == 0:
                    for r in right_set:
                        node = TreeNode(x)
                        node.right = r
                        res.append(node)
                        
                elif len(right_set) == 0:
                    for l in left_set:
                        node = TreeNode(x)
                        node.left = l
                        res.append(node)                
                else:
                    for l in left_set: 
                        for r in right_set:
                            node = TreeNode(x)
                            node.left = l
                            node.right = r
                            res.append(node)
            
            return res
        
        return generate(range(1,n+1))