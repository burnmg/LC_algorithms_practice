class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        f = lambda x,y: x*y
        return max(reduce(f, heapq.nlargest(3, nums)), reduce(f, heapq.nsmallest(2, nums) + heapq.nlargest(1, nums)))
