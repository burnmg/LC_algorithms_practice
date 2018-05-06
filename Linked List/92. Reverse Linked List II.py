# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        fake_head = ListNode(0)
        fake_head.next = head
        start = fake_head
        for i in range(m-1):
            start = start.next
            
        prev_node = None
        reverse_head = cur_node = start.next 
        for i in range(n-m+1):
            next_temp = cur_node.next
            cur_node.next = prev_node
            prev_node = cur_node
            cur_node = next_temp
        start.next = prev_node
        reverse_head.next = cur_node
        
        return fake_head.next
        
            
        
        