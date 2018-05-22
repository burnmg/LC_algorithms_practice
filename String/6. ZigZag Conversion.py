class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1: return s
        res = ['' for i in range(numRows)]
        movingDown = True
        i = 0
        step = -1
        for x in s:
            res[i] += x
            if i == len(res) - 1 or i == 0:
                step *= -1
                
            if movingDown:
                i += step
            else:
                i -= step
                
        return "".join(res)