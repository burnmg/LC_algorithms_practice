class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        low = 1
        high = len(nums)
        
        while low < high:
            mid = (low + high) // 2
            count = 0
            for x in nums:
                if x <= mid:
                    count += 1
            if count <= mid:
                low = mid + 1
            else:
                high = mid
        
        return low