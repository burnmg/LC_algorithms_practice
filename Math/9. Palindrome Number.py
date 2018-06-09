class Solution(object):
    
    def isPalindrome3(self, x):
        if x < 0:
            return False
        s = str(x)
        return s[::-1] == s
    
    def isPalindrome(self, x):
        if x < 0:
            return False
        if x < 10:
            return True
        
        left_divider = 1
        n = x
        while n>9:
            n //= 10 
            left_divider *= 10
        
        a = x
        while x >= 10:
            right = x % 10 
            left = a // left_divider
            if left != right:
                return False
            a = a % left_divider
            x = x // 10
            left_divider //= 10
            
        return True
    
    def isPalindrome3(self, x): # Fastest one 
        if x < 0:
            return False

        ranger = 1
        while x / ranger >= 10:
            ranger *= 10

        while x:
            left = x / ranger
            right = x % 10
            if left != right:
                return False
            
            x = (x % ranger) / 10
            ranger /= 100

        return True