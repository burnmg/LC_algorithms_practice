class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return nums[0]
        max_product_ending_i = min_product_ending_i = max_product = nums[0]
        for i in range(1, len(nums)):
            prev = [max_product_ending_i*nums[i], min_product_ending_i*nums[i] , nums[i]]
            max_product_ending_i = max(prev)
            min_product_ending_i = min(prev)
            max_product = max(max_product_ending_i, max_product)
        
        return max_product