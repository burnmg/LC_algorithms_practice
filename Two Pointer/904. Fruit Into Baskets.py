class Solution(object):
    def totalFruit(self, tree):
        """
        :type tree: List[int]
        :rtype: int
        """
        
        window_set = set([])
        
        count = 0
        max_count = 0
        prev_i = 0
        for i in range(len(tree)):
            window_set.add(tree[i])
            if len(window_set) > 2:
                window_set.remove(tree[prev_i-1])
                count = (i-prev_i)
            
            if i > 0 and tree[i] != tree[i-1]:
                prev_i = i  
            
            count += 1
            max_count = max(count, max_count)
            
        return max_count
