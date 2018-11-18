class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        prev_sums_dict = collections.defaultdict(int)
        prev_sums_dict[0] = 1
        
        cur_prefix_sum = 0
        count = 0
        for num in nums:
            cur_prefix_sum += num
            diff = cur_prefix_sum - k
            if diff in prev_sums_dict:
                count += prev_sums_dict[diff]
            
            prev_sums_dict[cur_prefix_sum] += 1
        
        return count