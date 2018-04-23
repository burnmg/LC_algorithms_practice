class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        s = set()
        for x in nums:
            if x in s:
                return True
            else:
                s.add(x)
        
        return False