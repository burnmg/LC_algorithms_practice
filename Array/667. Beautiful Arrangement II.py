class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        nums = [1] * n
        
        diff = k
        for i in range(1, k+1):
            if i % 2 == 1:
                nums[i] = nums[i-1] + diff
            else:
                nums[i] = nums[i-1] - diff
            diff -= 1
        for i in range(k+1, n):
            nums[i] = (i+1)
                
        return nums