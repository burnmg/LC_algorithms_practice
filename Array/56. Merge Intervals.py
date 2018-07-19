# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if len(intervals) <= 1:
            return intervals
        
        intervals.sort(key = lambda x:x.start)
        queue = collections.deque(intervals[1:])
        res = [intervals[0]]
        
        while queue:
            x = queue.popleft()
            if x.start <= res[-1].end:
                res[-1].end = max(res[-1].end, x.end)
            else:
                res.append(x)
            
        return res
            
        