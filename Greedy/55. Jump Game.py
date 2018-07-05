class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        furthest_destination = 0
        for i,n in enumerate(nums):
            
            if i > furthest_destination:
                return False
            furthest_destination = max(furthest_destination, i+n)
                
        return True