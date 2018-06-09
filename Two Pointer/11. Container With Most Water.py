class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        a, b = 0, len(height) - 1
        
        water = 0
        while a<b:
            water = max(water, min(height[a], height[b]) * (b-a))
            if height[a] < height[b]:
                a += 1
            else:
                b -= 1
        
        return water