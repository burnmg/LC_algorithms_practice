# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def splitListToParts(self, root, k):
        """
        :type root: ListNode
        :type k: int
        :rtype: List[ListNode]
        """
        
        # list len
        n = 0
        cur = root
        while cur:
            n += 1
            cur = cur.next
        
        # compute batch len
        base = n // k # 0
        remainder = n % k 
        
        # break the list
        res = []
        for i in range(k):
            res.append(root)
            count = base
            if root:
                if remainder > 0:
                    count += 1
                    remainder -= 1
                j = 0
                while j < count-1 and root: # count = 2
                    root = root.next
                    j += 1
                if root:
                    temp = root.next
                    root.next = None
                    root = temp
            
        
        return res
                
            