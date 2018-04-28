class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        res = []
        for h in range(0,12):
            for m in range(0, 60):
                if (bin(h) + bin(m)).count('1') == num:
                    res.append('%d:%02d' % (h,m))
        return res
    
        