# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def numComponents(self, head, G):
        """
        :type head: ListNode
        :type G: List[int]
        :rtype: int
        """
        g = set(G)
        comp = []
        cur = head
        prev_val = -1
        count = 0
        while cur:
            if cur.val in g and (len(comp) == 0 or comp[-1] == prev_val):
                comp.append(cur.val)
            else:
                if len(comp) > 0: 
                    count += 1
                comp = []
            prev_val = cur.val
            cur = cur.next
        if len(comp) > 0:
            count += 1
            
        
        return count