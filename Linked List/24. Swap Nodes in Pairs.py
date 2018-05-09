# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return head
        
        dum = ListNode(0)
        dum.next = head
        prev = dum
        cur = dum.next
        new_head = head.next

        while cur and cur.next:
            temp = cur.next
            cur.next = cur.next.next
            temp.next = cur
            prev.next = temp
            prev = cur
            cur = cur.next
        
        return new_head