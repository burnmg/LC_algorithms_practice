class Solution(object):
    def splitArray(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        def split(A):
            total = sum(A)
            for i in range(1, len(A)): A[i] += A[i-1]
            return {A[i-1] for i in range(1, len(A)-1) if A[i-1] == total - A[i]}
            
        return len(nums) > 6 and any(split(nums[:j]) & split(nums[j+1:]) \
                             for j in range(3, len(nums)-3))