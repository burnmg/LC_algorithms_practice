# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []
        self.traverse(root, self.stack)
    
    def traverse(self, root, stack):
        if not root:
            return
        
        self.traverse(root.right, stack)
        stack.append(root)
        self.traverse(root.left, stack)

    def hasNext(self):
        
        return len(self.stack) > 0
        

    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop().val
        

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())