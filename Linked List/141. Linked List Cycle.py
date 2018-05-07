class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        dum = ListNode(0)
        dum.next = head
        slow = dum
        fast = dum.next
        while fast and fast.next:
            if fast is slow: return True
            slow = slow.next
            fast = fast.next.next
        
        return False