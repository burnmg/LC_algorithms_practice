class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        x = 1
        temp = []
        res = []
        while True:
            if len(temp) == k:
                res.append(temp[:])
            if len(temp) == k or x > n - k + len(temp) + 1:
                if not temp:
                    return res
                x = temp.pop() + 1
            else:
                temp.append(x)
                x += 1
        