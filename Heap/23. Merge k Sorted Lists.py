# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        h = [(node.val, node) for node in lists if node]
        heapq.heapify(h)
        res = cur = ListNode(0)
        while h: 
            _, node = h[0]
            cur.next = node
            if node.next:
                heapq.heapreplace(h, (node.next.val, node.next))
            else:
                heapq.heappop(h)
            cur = cur.next
        
        return res.next
                