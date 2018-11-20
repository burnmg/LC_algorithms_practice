"""
# Definition for a Node.
class Node:
    def __init__(self, val, next):
        self.val = val
        self.next = next
"""
class Solution:
    def insert(self, head, insertVal):
        """
        :type head: Node
        :type insertVal: int
        :rtype: Node
        """
         # todo null case
        if not head:
            return ListNode(insertVal)

        # find end node and start node
        # if insertVal >= end node or insertVal <= start_node:
            # insert this val in the end        
        start_node = None
        end_node = None
        prev = head
        cur = head.next
        if not cur:
            head.next = ListNode(insertVal)
            head.next.next = head
                
        while True:
            if prev.val <= insertVal <= cur.val:
                prev.next = ListNode(insertVal)
                prev.next.next = cur
                return head
            if prev.val > cur.val:
                start_node = cur
                end_node = prev
            prev = cur 
            cur = cur.next
            if prev is head:
                break
            
        # if both are none. all values are equal. Insert on the head. 
        if not start_node:
            prev.next = ListNode(insertVal)
            prev.next.next = cur
        else:
            end_node.next = ListNode(insertVal)
            end_node.next.next = start_node
        
        return head