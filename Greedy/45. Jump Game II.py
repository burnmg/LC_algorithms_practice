class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        last_max_reach_point, current_max_reach_point = 0, 0
        step = 0
        for i in range(len(nums)-1):
            current_max_reach_point = max(i + nums[i], current_max_reach_point)
            if i == last_max_reach_point:
                last_max_reach_point = current_max_reach_point
                step += 1
        
        return step
            
            
        