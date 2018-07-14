class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        a,b = 0,len(s) - 1
        
        has_one_more_chance = True
        
        while a<b:
            
            if s[a] != s[b]:
                return self.isValid(s[a+1:b+1]) or self.isValid(s[a:b]) 
            a += 1
            b -= 1
        
        return True
    
    def isValid(self, s):
        a,b = 0,len(s) - 1
        
        while a<b:
            
            if s[a] != s[b]:
                return False
            a += 1
            b -= 1
        
        return True        