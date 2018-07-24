# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        return self.merge_sort(head)
    
    def merge_sort(self, head):
        
        if not head:
            return None
        
        if not head.next:
            return head
        
        dummy = ListNode(0)
        slow, fast, slow_prev = head, head, dummy
        
        # partition
        while fast and fast.next:
            fast = fast.next.next
            slow_prev = slow
            slow = slow.next
        slow_prev.next = None
        
        cur1 = self.merge_sort(head)
        cur2 = self.merge_sort(slow)
        
        dummy = ListNode(0)
        cur = dummy
        
        while cur1 and cur2:
            if cur1.val < cur2.val:
                cur.next = cur1
                cur1 = cur1.next
            else:
                cur.next = cur2
                cur2 = cur2.next
            cur = cur.next
        
        if cur1:
            cur.next = cur1
        
        if cur2:
            cur.next = cur2
            
        return dummy.next
        
        
        
                
                
                
        