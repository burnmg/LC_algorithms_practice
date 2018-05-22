class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        d = collections.Counter(s)
        l = []
        for k in d:
            l.append((d[k], k))
        sorted_l = sorted(l, key=(lambda x:x[0]), reverse = True)
        
        return "".join([x[1] * x[0] for x in sorted_l])