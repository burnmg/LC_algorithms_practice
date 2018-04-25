# Ruolin's Leetcode Practice NoteBook
A repo for Ruolin's LC practice

## Array

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)
```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [1] * len(nums)
        p = nums[0]
        for i in range(1, len(nums)):
            res[i] = p
            p *= nums[i]
        
        p = nums[-1]
        for i in range(len(nums)-2, -1, -1):
            res[i]  *= p
            p *= nums[i]
                
        return res
```
Calculate products from start to end and then from end to start without considering `n[i]` at `i` position
### [667. Beautiful Arrangement II](https://leetcode.com/problems/beautiful-arrangement-ii/description/)
```python
class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        nums = [1] * n
        
        diff = k
        for i in range(1, k+1):
            if i % 2 == 1:
                nums[i] = nums[i-1] + diff
            else:
                nums[i] = nums[i-1] - diff
            diff -= 1
        for i in range(k+1, n):
            nums[i] = (i+1)
                
        return nums
```
We firstly satisfy the condition of k different differences. Then we append rest of numbers.

## Two Pointer
### [27. Remove Element](https://leetcode.com/problems/remove-element/description/)
```python
class Solution(object):
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        i,j = 0, len(nums) - 1
        while i <= j:
            if nums[i] == val:
                nums[i],nums[j] = nums[j],nums[i]
                j -= 1
            else:
                i += 1
        return i
```
Remember to use '<=' in the loop of two pointer algorithms. 
## Hashtable
### [697. Degree of an Array](https://leetcode.com/problems/degree-of-an-array/description/)
```python
class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start,end = {},{}
        for i,x in enumerate(nums):
            start.setdefault(x,i)
            end[x] = i
        c = collections.Counter(nums)
        degree = max(c.values())
        
        return min(end[x] - start[x] + 1 for x in c if c[x] == degree)

```
`dict.setdefault(x,i)` adds an element only if they key does not exist.

`collections.Counter(nums)` returns a dict that counts elements' frequency. Each entry is `{'element':count}`

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

