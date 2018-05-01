class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        tail = 0
        smaller_product_before_i = nums[0]
        count = 0 if nums[tail] >= k else 1
        for i in range(1, len(nums)):
            p = smaller_product_before_i * nums[i]
            while p>=k and tail<=i:
                p = p / nums[tail]
                tail += 1
            count += i - tail + 1
            smaller_product_before_i = p
        
        return count