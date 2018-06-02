class Solution(object):

    def findKthLargest(self, nums, k):
        return heapq.nlargest(k, nums)[k-1]