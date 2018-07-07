class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        top_three = []
        for x in nums:
            if x not in top_three:
                top_three.append(x)
            if len(top_three) > 3:
                top_three.sort(reverse = True)
                top_three.pop()
        
        if len(top_three) < 3:
            return max(top_three)
        else:
            top_three.sort(reverse = True)
            return top_three[2]
            
            
                
        