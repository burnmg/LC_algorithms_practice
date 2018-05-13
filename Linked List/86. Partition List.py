# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        dum = ListNode(0)
        dum.next = head
        cur = head
        prev = dum
        temp_list = ListNode(0)
        cur2 = temp_list 

        while cur:
            if cur.val >= x:
                prev.next = cur.next
                cur2.next = cur
                cur2 = cur2.next
                cur = cur.next
                cur2.next = None 
            else:
                prev = cur
                cur = cur.next
        
        prev.next = temp_list.next


        return dum.next
                