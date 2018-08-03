# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        left_height = self.count_height(root.left)
        right_height = self.count_height(root.right)
        
        node_count = 1
        if left_height == right_height:
            node_count += self.count_node_of_complete_tree(left_height) + self.countNodes(root.right)
        else:
            node_count += self.count_node_of_complete_tree(right_height) + self.countNodes(root.left)
        
        return node_count
            
    def count_node_of_complete_tree(self, height):
        
        return sum(pow(2, i) for i in range(height))
            
    def count_height(self, root):
        
        if not root:
            return 0
        
        height = 1
        while root.left:
            root = root.left
            height += 1
        return height