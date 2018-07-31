# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        if len(intervals) <= 1:
            return len(intervals)
        
        intervals.sort(key = lambda x:x.start)
        
        cur_rooms = 1
        recent_end_time = [intervals[0].end]
        
        for i in range(1, len(intervals)):
            if intervals[i].start >= recent_end_time[0]:
                heapq.heappushpop(recent_end_time, intervals[i].end)
            else:
                cur_rooms += 1
                heapq.heappush(recent_end_time, intervals[i].end)
                
        return cur_rooms