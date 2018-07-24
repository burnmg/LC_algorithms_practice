# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head: return None
        
        dum = ListNode(0)
        dum.next = head
        dele = False
        prev = dum
        cur = head
        
        while cur.next:
            if cur.val == cur.next.val:
                prev.next = cur.next
                cur = prev.next
                dele = True
            elif dele:
                prev.next = cur.next
                dele = False
                cur = cur.next
            else:
                prev = prev.next
                cur = cur.next
        if dele:
            prev.next = cur.next
        
        return dum.next
                

                
        