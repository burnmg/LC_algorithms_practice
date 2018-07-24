"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""
class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return None
        
        
        stack = [head]
        temp_head = Node(0, None, None, None)
        prev = temp_head
        
        while stack:
            node = stack.pop()
            if node.next:
                stack.append(node.next)
            if node.child:
                stack.append(node.child)
            
            
            node.next = None
            node.prev = prev
            node.child = None
            
            prev.next = node
            prev = node
        
        temp_head.next.prev = None # Make sure the real head does not have a "prev"
        return temp_head.next
        
            
            