# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ''
        stack = [root]
        res = []
        while stack:
            x = stack.pop()
            res.append(str(x.val))
            
            if x.right:
                stack.append(x.right)
            if x.left:
                stack.append(x.left)
        
        return ' '.join(res)
                
            
            

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        queue = collections.deque(int(s) for s in data.split())
        def build(min_val, max_val):
            if queue and min_val < queue[0] < max_val:
                root = TreeNode(queue.popleft())
                root.left = build(min_val, root.val)
                root.right = build(root.val, max_val)
                return root
            return None
        return build(float('-inf'), float('inf'))
                

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))