class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0
        
        left_maxs = [height[0]] * len(height)
        
        for i in range(1, len(height)):
            left_maxs[i] = max(left_maxs[i-1], height[i])
        
        right_maxs = [height[-1]] * len(height)
        
        for i in range(len(height) - 2, -1, -1):
            right_maxs[i] = max(right_maxs[i+1], height[i])
        
        water = 0
        for i in range(len(height)):
            water += min(left_maxs[i], right_maxs[i]) - height[i]

        return water