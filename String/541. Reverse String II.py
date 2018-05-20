class Solution(object):
    def reverseStr2(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        sl = list(s)
        i = 0
        while i < len(sl):
            a,b = i, min(len(sl), i+k)-1

            while a<b:
                sl[a],sl[b] = sl[b], sl[a]
                a += 1
                b -= 1
            i += 2*k
        
        return "".join(sl)
    
    def reverseStr(self, s, k): # Use Python library
        s = list(s)
        for i in xrange(0, len(s), 2*k):
            s[i:i+k] = (s[i:i+k])[::-1]
        return "".join(s)