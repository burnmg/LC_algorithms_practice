class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        if num == 0: return ["0:00"]
        one_bits = range(0, num)
        max_pos = 10
        res = []
        for i in range(len(one_bits)-1, -1, -1):
            for j in range(one_bits[i], max_pos+1):
                one_bits[i] = j
                bin = [0] * 10
                for k in one_bits:
                    bin[k] = 1
                h = int("".join(map(str, bin[:4])), 2)
                m = int("".join(map(str, bin[4:])), 2)
                res.append('%d:%02d' % (h, m))
                
            max_pos -= 1
                
                
        return res
        
        