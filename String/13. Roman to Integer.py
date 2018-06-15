class Solution(object):
    def romanToInt2(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        i = len(s) - 1
        while i >= 0:
            if s[i] == 'I':
                res += 1
            elif s[i] == 'V':
                res += 5
                if i-1 >= 0 and s[i-1] == 'I':
                    res -= 1
                    i -= 1
            elif s[i] == 'X':
                res += 10
                if i-1 >= 0 and s[i-1] == 'I':
                    res -= 1
                    i -= 1
            elif s[i] == 'L':
                res  += 50
                if i-1 >= 0 and s[i-1] == 'X':
                    res -= 10
                    i -= 1
            elif s[i] == 'C':
                res += 100
                if i-1 >= 0 and s[i-1] == 'X':
                    res -= 10
                    i -= 1                
            elif s[i] == 'D':
                res += 500
                if i-1 >= 0 and s[i-1] == 'C':
                    res -= 100
                    i -= 1                
            elif s[i] == 'M':
                res += 1000
                if i-1 >= 0 and s[i-1] == 'C':
                    res -= 100
                    i -= 1                    
            else:
                return -1
            i -= 1
        return res
    
    def romanToInt(self, s):
        _map = {'I':1,
               'V':5,
               'X':10,
               'L':50,
               'C':100,
               'D':500,
               'M':1000}
        
        res = _map[s[-1]]
        
        for i in xrange(len(s) - 2, -1 , -1) :
            res = res + _map[s[i]] if _map[s[i]] >= _map[s[i+1]] else res - _map[s[i]]
        
        return res
        
        
        
        
        
        
        