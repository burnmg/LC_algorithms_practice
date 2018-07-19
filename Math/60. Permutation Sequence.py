class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        
        return self.permutation(range(1, n+1), k, math.factorial(n))
        
    def permutation(self, nums, k, factorial):
        
        if len(nums) == 1:
            return str(nums[0])
        start_digit = (k-1) // (factorial / len(nums))
        
        return str(nums[start_digit]) + self.permutation(nums[:start_digit] + nums[start_digit+1:],
                                                         1 + (k-1) % (factorial / len(nums)), 
                                                        factorial / len(nums))
        
        