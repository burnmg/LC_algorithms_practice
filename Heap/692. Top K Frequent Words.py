class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # count frequencies of each word
        word_freq = collections.Counter(words)
       
        # build a max heap with k nodes
        heap = []
        
        for key in word_freq:
        # insert a node into the heap. We have two criteria
            # word_freq
            # alphabetical order
            heapq.heappush(heap, MyTuple(key, -word_freq[key]))
        
        # return k element from the heap
        return [heapq.heappop(heap).key for i in range(k)]

# Use this as a helper
class MyTuple:
    def __init__(self, key, count):
        self.key = key
        self.count = count
        
    def __lt__(self, other):
        
        if self.count != other.count:
            return self.count < other.count
        else:
            return self.key < other.key
        
