# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dum = ListNode(0)
        dum.next = head
        before_del = p = dum
        
        for i in range(n+1):
            p = p.next
        while p:
            before_del = before_del.next
            p = p.next
        
        before_del.next = before_del.next.next
        
        return dum.next