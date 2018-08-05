class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        freqs = collections.Counter(tasks).values()
        
        max_freq = max(freqs)
        num_max_freqs = freqs.count(max_freq)
        parts = max_freq - 1
        num_empty_slots = parts * (n - (num_max_freqs - 1))
        
        return max(num_empty_slots + num_max_freqs * max_freq,  sum(freqs))