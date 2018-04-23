# Ruolin's Leetcode Practice NoteBook
A repo for Ruolin's LC practice

## Backtracking
### [526. Beautiful Arrangement](https://leetcode.com/problems/beautiful-arrangement/description/)
```python
class Solution(object):
    def countArrangement(self, N):
        """
        :type N: int
        :rtype: int
        """
        def count(i,X):
            if i == 1:
                return 1
            
            return sum(count(i-1,X-{x}) for x in X if x % i == 0 or i % x == 0)
        return count(N, set(range(1,N+1)))
                
```
Trick: lower ith is likely to contain "beautiful" value, so we run backwards so we can eliminate "unbeautiful" chains faster. This makes our search space smaller. 
