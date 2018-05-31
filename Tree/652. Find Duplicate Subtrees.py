# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def findDuplicateSubtrees(self, root): 
        
        def hash_tree(root):
            if not root:
                return None
            tree_id = (hash_tree(root.left), root.val, hash_tree(root.right))
            count[tree_id].append(root)
            return tree_id
        
        count = collections.defaultdict(list)
        hash_tree(root)
        return [nodes.pop() for nodes in count.values() if len(nodes) >= 2]
        
        
            