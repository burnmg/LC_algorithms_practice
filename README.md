   
            
# Ruolin's Leetcode Practice NoteBook
Practice makes perfect.

## Array

### [229. Majority Element II](https://leetcode.com/problems/majority-element-ii/description/)

```python
class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        candidate1, candidate2, count1, count2 = 0, 1, 0, 0
        
        for x in nums: 
            if x == candidate1:
                count1 += 1
            elif x == candidate2:
                count2 += 1
            elif count1 == 0:
                candidate1 = x
                count1 = 1
            elif count2 == 0:
                candidate2 = x
                count2 = 1
            else:
                count1 -= 1
                count2 -= 1
        return [x for x in [candidate1, candidate2] if nums.count(x) > len(nums) // 3]
```
See [here](https://leetcode.com/problems/majority-element-ii/discuss/63520/Boyer-Moore-Majority-Vote-algorithm-and-my-elaboration). 

Use majority vote method. The original algorithm works for one candidate and count. This is extension to 2 candidates. We only reduce two counts if none of them are matched to `x`. Otherwise, we increment one of them without reducing the count of the other. 

### [324. Wiggle Sort II](https://leetcode.com/problems/wiggle-sort-ii/description/)


```python
class Solution:
    def wiggleSort(self, nums):
        nums.sort()
        half = len(nums[::2])
        nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1] # Check this pythonic opertation

class Solution: # Easy Version
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        sorted_nums = sorted(nums)
        
        
        j = 1
        while j < len(nums):
            nums[j] = sorted_nums.pop()
            j += 2        
        
        j = 0
        while j < len(nums):
            nums[j] = sorted_nums.pop()
            j += 2
        

``` 

Sort the list first.

Them put first half in odd indices and second half to even indices.


### [243. Shortest Word Distance](https://leetcode.com/problems/shortest-word-distance/description/)

```python
class Solution(object):
    def shortestDistance(self, words, word1, word2):
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        min_dist = len(words)
        index1, index2 = len(words), len(words)
        
        for i in range(len(words)):
            if words[i] == word1:
                index1 = i
                min_dist = min(min_dist, abs(index1-index2))
            elif words[i] == word2:
                index2 = i
                min_dist = min(min_dist, abs(index1-index2))                
                
        return min_dist
    
```
### [244. Shortest Word Distance II](https://leetcode.com/problems/shortest-word-distance-ii/description/)

```python
class WordDistance(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.words = words
        self.memo = {}
        self.indices = collections.defaultdict(list)
        for i, x in enumerate(words):
            self.indices[x].append(i)
        


    def shortest(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        min_dist = len(self.words)
        indices1, indices2 = self.indices[word1], self.indices[word2]
        i, j = 0, 0
        while i < len(indices1) and j < len(indices2):
            min_dist = min(min_dist, abs(indices1[i]-indices2[j]))
            if indices1[i] < indices2[j]:
                i += 1
            else:
                j += 1

        return min_dist
        


# Your WordDistance object will be instantiated and called as such:
# obj = WordDistance(words)
# param_1 = obj.shortest(word1,word2)
```
Prestore all indices first. 

The prestored indices are sorted for each word.

Then use two-pointer way to iterate them. On each iteration, we attempt to reduce the distance. 


### [31. Next Permutation](https://leetcode.com/problems/next-permutation/description/)


```python
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1: return

        suffix_head = 0
        for i in range(len(nums) - 1, 0, -1):
            if nums[i - 1] < nums[i]:
                suffix_head = i
                for j in range(len(nums) - 1, suffix_head - 1, -1):
                    if nums[suffix_head - 1] < nums[j]:
                        temp = nums[suffix_head - 1]
                        nums[suffix_head - 1] = nums[j]
                        nums[j] = temp
                        break
                break
        
        # reverse suffix
        start,end = suffix_head, len(nums) - 1
        while start < end:
            temp = nums[start]
            nums[start] = nums[end]
            nums[end] = temp
            end -= 1
            start += 1
    
```

### [73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/description/)

```python
class Solution(object):
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        
        m = len(matrix)
        if m == 0:
            return
        n = len(matrix[0])
        if n == 0:
            return
        

        zero_top_row = False
        for j in range(n):
            if matrix[0][j] == 0:
                zero_top_row = True
                break
                
        zero_left_col = False 
        for i in range(m):
            if matrix[i][0] == 0:
                zero_left_col = True
                break
        
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0
                    
        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0
        
        if zero_top_row:
            for j in range(n):
                matrix[0][j] = 0
        
        if zero_left_col:
            for i in range(m):
                matrix[i][0] = 0
        
        
            
                    
```
Trick: Use the first col and first row as a memory to store "zero" columns and rows. This gives us O(1) space complexity. 


[Explanation](https://leetcode.com/problems/next-permutation/discuss/13866/Share-my-O(n)-time-solution)

### [48. Rotate Image](https://leetcode.com/problems/rotate-image/description/)

```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        def rec_rotate(matrix, start, end):
            if start >= end:
                return
            for i in range(start, end):
                
                 matrix[i][end], matrix[end][end-i+start], matrix[end-i+start][start], matrix[start][i] = matrix[start][i], matrix[i][end], matrix[end][end-i+start], matrix[end-i+start][start]
            rec_rotate(matrix, start+1, end-1)
                
        rec_rotate(matrix, 0, len(matrix) - 1)
```

### [152. Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/)
```python
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return nums[0]
        max_product_ending_i = min_product_ending_i = max_product = nums[0]
        for i in range(1, len(nums)):
            prev = [max_product_ending_i*nums[i], min_product_ending_i*nums[i] , nums[i]]
            max_product_ending_i = max(prev)
            min_product_ending_i = min(prev)
            max_product = max(max_product_ending_i, max_product)
        
        return max_product
                                        
```
Template for subarray DP problem. 

Use the though of `max_ending_here` DP. It is a similar problem to "Max sub array". 

### [119. Pascal's Triangle II](https://leetcode.com/problems/pascals-triangle-ii/description/)
```python
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        row = [1]
        for i in range(rowIndex):
            line1 = row + [0]
            line2 = [0] + row
            zipped = zip(line1, line2)
            row = map(sum, zipped)
        return row

```
VERY GOOD PROBLEM.
Use 0 as a padding number for each row.

Use `zip()` to compose each numbers from previous row. Very smart.

### [238. Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/description/)
```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [1] * len(nums)
        for i in range(1, len(nums)):
            res[i] = nums[i - 1] * res[i-1]
        
        prev = nums[-1]
        for i in range(len(nums)-2, -1, -1):
            res[i] = res[i] * prev
            prev *= nums[i]
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

### [Fruit Into Basket]()
```python
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
                window_set.remove(tree[prev_i-1]) #  one-shot shrink the windo
                count = (i-prev_i) 
            
            if i > 0 and tree[i] != tree[i-1]:
                prev_i = i  
            
            count += 1
            max_count = max(count, max_count)
            
        return max_count

```
Two pointer approach. It is further optimized with one-shot shrink the window


### [11. Container With Most Water](https://leetcode.com/problems/container-with-most-water/description/)

```python 
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        a, b = 0, len(height) - 1
        
        water = 0
        while a<b:
            water = max(water, min(height[a], height[b]) * (b-a))
            if height[a] < height[b]:
                a += 1
            else:
                b -= 1
        
        return water

```
Good question. 

For each step, we remove the shorted wall. 

Given two walls `a` and `b`. If the are walls 'c' and 'd' between `a` and `b` (inclusive), then `c` and `d` will be reached by the algorithm. This can be proved by induction. 

### [15. 3Sum](https://leetcode.com/problems/3sum/description/)
```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]: # handle duplicate 
                continue
            l,r = i+1,len(nums)-1
            while l< r :
                s = nums[i] + nums[l] + nums[r]
                if s == 0:
                    while l< r and nums[l] == nums[l+1]: # handle duplicate 
                        l +=1
                    while l< r and nums[r] == nums[r-1]: # handle duplicate 
                        r -= 1
                    res.append([nums[i] , nums[l] , nums[r]]) 
                    r -= 1
                    l += 1
                elif s < 0:
                    l += 1
                else:
                    r -= 1
                
        return res

            
```
See how to handle the duplicate case. 



### [18. 4Sum](https://leetcode.com/problems/4sum/description/)

```python 
class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        nums.sort()
        results = []
        self.NSum(nums, target, 4, [], results)
        
        return results
        
    def NSum(self, nums, target, N, result, results):
        if len(nums) < N or N < 2:
            return
        
        if N == 2:
            l,r = 0, len(nums) - 1
            while l < r:
                s = nums[l] + nums[r] 
                if s == target:
                    
                    results.append(result + [nums[l], nums[r]])
                    r -= 1
                    l += 1
                    while l < r and nums[l] == nums[l-1]: # Remove the replicate results
                        l += 1 
                    while l < r and nums[r] == nums[r+1]:  # Remove the replicate results
                        r -= 1
                elif s < target:
                    l += 1
                else:
                    r -= 1
            
        else:
            for i in range(len(nums)-N+1):
                if target < N*nums[i] or target > N*nums[-1]: # Avoid unnecessary search. 
                    break
                if i == 0 or i > 0 and nums[i-1] != nums[i]:
                    self.NSum(nums[i+1:], target-nums[i], N-1, result+[nums[i]], results)
                    
        return 
                    
                
            
```
Template. Very good solution 
Sort the list, then reduce the problem to 2Sum by backtracking. 




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

### [347. Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)

```python 
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        res = collections.Counter(nums).most_common(k)
        
        return [x[0] for x in res]
```
Use `most_common` after `Counter`

### [454. 4Sum II](https://leetcode.com/problems/4sum-ii/description/)
```python 
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        AB = collections.Counter(a+b for a in A for b in B)
        
        return sum(AB[-c-d] for c in C for d in D)
        
```
Remember of usage of `collections.Counter`

The idea is to use hashtable and count all possible sum (a+b)values of AB elements, and then find all negative sums (-c-d) of CD that equal to the hashed sum of AB. For each time we find a match `AB[-c-d]`, we add the value of `AB[-c-d]` to our returned count. 

### [680. Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/description/)

```python
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        a,b = 0,len(s) - 1
        
        has_one_more_chance = True
        
        while a<b:
            
            if s[a] != s[b]:
                return self.isValid(s[a+1:b+1]) or self.isValid(s[a:b]) # run two sub-calls to isValid by removing each sides' ends. 
            a += 1
            b -= 1
        
        return True
    
    def isValid(self, s):
        a,b = 0,len(s) - 1
        
        while a<b:
            
            if s[a] != s[b]:
                return False
            a += 1
            b -= 1
        
        return True        
```

### [283. Move Zeroes](https://leetcode.com/problems/move-zeroes/description/)

```python 
class Solution(object):
# in-place
    def moveZeroes(self, nums):
        zero = 0  # records the position of "0"
        for i in xrange(len(nums)):
            if nums[i] != 0:
                nums[i], nums[zero] = nums[zero], nums[i]
                zero += 1
```

### [560. Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/description/)

```python
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        cur_sum = 0
        count = 0
        _hash = {0:1} # hash 0 elements' sum
        for i in range(len(nums)):
            cur_sum += nums[i]
            diff = cur_sum - k
            if diff in _hash:
                count += _hash[diff]
                
            if cur_sum in _hash:
                _hash[cur_sum] += 1
            else:
                _hash[cur_sum] = 1
                
        return count
                                     
```
We want to find subarrays that are added to `k`. 

Use hash table to store all cumulative sums, and then use `cur_sum[j] - cur_sum[i]` to find sums ending at `j`. Then for any `j`, if there are `y` `cur_sum[i]` such that `cur_sum[j] = cur_sum[i] - k`, then there will be `y` sums ending at `j` equals to k. 


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

### Contains Duplicate II
```python
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        d = {}
        for i,x in enumerate(nums):
            if x in d and i-d[x] <= k:
                    return True
            d[x] = i
        
        return False
```
## Dynamic Programming

### [968. Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minCameraCover(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        def dp(node):
            """
            return have_camera_no_need, no_camera_no_need, no_camera_need
            """
            
            if not node:
                return float("inf"), 0, float("inf")
            
            if not node.left and not node.right:
                return 1, float("inf"), 0
            
            left_res = dp(node.left)
            right_res = dp(node.right)
            
            have_camera_no_need = 1 + min(left_res) + min(right_res)
            no_camera_no_need = min(left_res[0] + min(right_res[:2]), right_res[0] + min(left_res[:2]))
            no_camera_need = left_res[1] + right_res[1]
            
            return have_camera_no_need, no_camera_no_need, no_camera_need
        
        return min(dp(root)[:2])

```
Good 3D DP problem. 

### [727. Minimum Window Subsequence](https://leetcode.com/problems/minimum-window-subsequence/)

```python
class Solution(object):
    def minWindow(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        n = len(S)
        m = len(T)
        
        dic = dict()
        for i, s in enumerate(T):
            dic.setdefault(s, []).append(i)
            
        dp = [-1 for i in xrange(m)]
        
        count = n+1
        start = -1
        
        for index, c in enumerate(S):
            if c in dic:
                for i in dic[c][::-1]:
                    if i == 0:
                        dp[i] = index
                    else:
                        dp[i] = dp[i-1]
                    if i == m-1 and dp[i] >= 0 and index - dp[i]+1 < count:
                        count = index - dp[i] + 1
                        start = dp[i]
        if dp[-1] < 0:
            return ""
        return S[start:start+count]
```
Google question. 
This answer is so brilliant.

The idea is too maintain a `dp`. `dp[i]` tracks the starting positions of subsequence in `S` that matches until `i`th char in `T`. 

### [568. Maximum Vacation Days](https://leetcode.com/problems/maximum-vacation-days/description/)
```python
class Solution(object):
    def maxVacationDays(self, flights, days):
        """
        :type flights: List[List[int]]
        :type days: List[List[int]]
        :rtype: int
        """
        # preprocess the flights
        for i in range(len(flights)):
            flights[i][i] = 1

        cur_vocations = [days[i][0] if flights[0][i] != 0 else float('-inf') for i in range(len(days))]
        longest_voc = max(cur_vocations)

        for i in range(1, len(days[0])):  # for each day
            new_vocations = []
            for j in range(len(flights)):  # for each arriving place. j is the new place. k is the previous place
                new_vocations.append(max(
                    [days[j][i] + cur_vocations[k] if flights[k][j] == 1 else float('-inf') for k in range(len(flights))]))
                longest_voc = max(longest_voc, new_vocations[j])
            cur_vocations = new_vocations
        return longest_voc


```



### [418. Sentence Screen Fitting](https://leetcode.com/problems/sentence-screen-fitting/description/)

```python
class Solution(object):
    def wordsTyping(self, sentence, rows, cols):
        s = ' '.join(sentence) + ' '
        count = 0
        for i in xrange(rows):
            count += cols
            if s[count % len(s)] == ' ':
                count += 1
            else:
                while count > 0 and s[ (count - 1) % len(s) ] != ' ':
                    count -= 1
        return count/ len(s)
```
https://leetcode.com/problems/sentence-screen-fitting/discuss/90869/Python-with-explanation

### [368. Largest Divisible Subset](https://leetcode.com/problems/largest-divisible-subset/description/)

```python
class Solution:
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) == 0:
            return []
        
        dp = []
        nums.sort()
        
        for n in nums:
            cur_set = [n]
            for _set in dp:
                if n % _set[-1] == 0 and len(_set) + 1 > len(cur_set):
                    cur_set = _set + [n]
            
            dp.append(cur_set)
        
        return max(dp, key = lambda x: len(x))
                    
```

### [486. Predict the Winner](https://leetcode.com/problems/predict-the-winner/description/)

```python
class Solution:
    def PredictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        memo = {}
    
        def helper(_sum, nums, start, end):
            
            if (start, end) in memo:
                return memo[(start, end)]
            
            if start == end:
                memo[(start, end)] = nums[start]
                return nums[start]
            
            score1 = _sum - helper(_sum - nums[start], nums, start+1, end) 
            score2 = _sum - helper(_sum - nums[end], nums, start, end-1)
            
            res = max(score1, score2)
            memo[(start, end)] = res
            
            return res
        
        return helper(sum(nums), nums, 0, len(nums)-1) * 2 >= sum(nums)
```
In each recursive step, we compute two different possible moves: pick first number or last number.
If we pick the first number, the maximal possible score I can get is the the sum of all numbers subtracting the maximal possible sore the other player can get in `nums[1:]` if he is the first-mover in `nums[1:]`, because we assume each player is aplying the game optimally. 

Similarly, If we pick the last number, the maximal possible score I can get is the the sum of all numbers subtracting the maximal possible sore the other player can get in `nums[:len(nums)-1]` if he is the first-mover in `nums[:len(nums)-1]`.

Above is the naive solution that will exceed time limit. 

My code then uses Memoization by hashing sub-solution for each subarray. With memoization, my algorithm beats 90% of runtimes. 


### [823. Binary Trees With Factors](https://leetcode.com/problems/binary-trees-with-factors/description/)

```python
class Solution:
    def numFactoredBinaryTrees(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        A.sort()
        dp = {}
        
        for i in range(len(A)):
            dp[A[i]] = 1
            for j in range(i):
                if A[i] % A[j] == 0 and A[i] / A[j] in dp:
                    dp[A[i]] += dp[A[i] / A[j]] * dp[A[j]]
        
        return sum(dp.values()) % (10**9 + 7)
```
Find a number that is one product of the current number. Then we see if the other number is in our dp list. If yest, simply add the product of their dp values. 

### [42. Trapping Rain Water][https://leetcode.com/problems/trapping-rain-water/description/]

```python
class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0
        
        left_maxs = [height[0]] * len(height)
        
        for i in range(1, len(height)):
            left_maxs[i] = max(left_maxs[i-1], height[i])
        
        right_maxs = [height[-1]] * len(height)
        
        for i in range(len(height) - 2, -1, -1):
            right_maxs[i] = max(right_maxs[i+1], height[i])
        
        water = 0
        for i in range(len(height)):
            water += min(left_maxs[i], right_maxs[i]) - height[i]

        return water
```
Scan from left and then to right and stores all max wall heights.  

The shared spaces minus wall spaces are water levels. 

See Solution 2 [here](https://leetcode.com/problems/trapping-rain-water/solution/)

### [97. Interleaving String](https://leetcode.com/problems/interleaving-string/description/)

```python
class Solution(object):
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        if len(s1) + len(s2) != len(s3):
            return False
        
        stack = [(0,0)]
        visited = set([(0,0)])
        
        while stack:
            coo = stack.pop() # coordinate
            
            if coo[0] == len(s1) and coo[1] == len(s2):
                return True      
            
            down = (coo[0]+1, coo[1]) 
            if down[0]-1 < len(s1) and down not in visited and s1[down[0]-1] == s3[down[0] + down[1] -1]: # move rightward on the matrix. Use '-1' converts the matrix coordiante to string's index.
                visited.add(down)
                stack.append(down)
            
            right = (coo[0], coo[1]+1)
            if right[1]-1 < len(s2) and right not in visited and s2[right[1]-1] == s3[right[0] + right[1] -1]: # move downward on the matrix 
                visited.add(right)
                stack.append(right)
        
        return False
```
Very good problem. Template for Dynamic Programming with a 2D matrix. 

In this problem, the basic DP solution is to use a 2D matrix. 

(Note that we add 'empty' in front of s1, s2, s3)

Each entry `M[i,j]` on the matrix is whether the `s3[i+j]` is the interleave of `s1[i]` and `s2[j]`. Note that we add have prefixes 'Empty' `s1`,`s2`,`s3`

DP process: 
* `M[i+1,j]` is true when `M[i,j]` is `True` and `M[i+1,j] == s1[i+1]` is `True`. (Move downwards on the matrix)
* `M[i,j+1]` is true when `M[i,j]` is `True` and `M[i,j+1] == s2[i+1]` is `True`. (Move rightwards on the matrix)

My code is optimised as a DFS solution so we don't compute the whole matrix and we don't maintain a matrix in the memoery. We only search for the `True` entries on the matrix. 

A video demo is [here](https://www.youtube.com/watch?v=ih2OZ9-M3OM). 

### [132. Palindrome Partitioning II](https://leetcode.com/problems/palindrome-partitioning-ii/description/)

```python
class Solution(object):
    def minCut(self, s):
        
        cut = [x for x in range(-1, len(s))]
        
        for i in range(1, len(s)):
            for j in range(i+1):
                if s[j:i+1] == s[j:i+1][::-1]:
                    cut[i+1] = min(cut[i+1], cut[j]+1)
        
        return cut[-1]
```
Example of O(n^2) dynamic programming. 

Hard problem. 

### [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/description/)

```python
class Solution(object):
    def longestPalindrome2(self, s): # my slower solution
        """
        :type s: str
        :rtype: str
        """
        
        max_sub_str = ''
        for i in range(len(new_s)):
            a, b = i, i+1
            length = 1
            
            while a>= 0 and b < len(new_s) and new_s[a] == new_s[b]:
                length += 2
                a -= 1
                b += 1
            if length > len(max_sub_str):
                max_sub_str = length
                max_sub_str = new_s[a+1:b]
        
        return max_sub_str
    
    def longestPalindrome(self, s): 
        if len(s) == 0:
            return 0
        
        max_len = 1
        max_end = 0
        for i in range(1, len(s)):
            start = (i - max_len) 
            if s[start:i+1] == s[start:i+1][::-1]:
                max_len += 1
                max_end = i
            elif start >= 1 and s[start-1:i+1] == s[start-1:i+1][::-1]:
                max_len += 2
                max_end = i
                
        return s[max_end-max_len+1:max_end+1]
                
    
    
     
```
The fast solution uses the proved rule: let M be the longestPalindrome of `s[:i-1]`, then `s[:i]`'s longestPalindrome can only be M, M+1 or M+2. 

So we only needs to examine whether substrings ending at `s[i]` with length M+1 and M+2 are palindrome. 

## [213. House Robber II](https://leetcode.com/problems/house-robber-ii/description/)

```python
class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        return max(self.dp_single_path(nums[1:]), self.dp_single_path(nums[:len(nums)-1]))
    
    def dp_single_path(self, nums):
        
        prev_prev_max, prev_max = nums[0], max(nums[0], nums[1])
        cur_max = prev_max
        
        for i in range(2, len(nums)):
            cur_max = max(nums[i] + prev_prev_max, prev_max)
            prev_prev_max = prev_max
            prev_max = cur_max
        
        return prev_max
```
Find max money in the circle with `n` houses, we just need the max of 1 to `n-1` housrs and the 2 to `n` houses. 

### [91. Decode Ways](https://leetcode.com/problems/decode-ways/description/)

```python 
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s[0] == '0':
            return 0
        if len(s) == 1 or len(s) == 0:
            return len(s)
        
        prev_res = 1
        if s[1] == '0':
            if s[0] == '1' or s[0] == '2':
                prev_res = 1
            else:
                return 0
        elif int(s[0:2]) < 27:
            prev_res = 2
        
        prev_prev_res = 1
        
        for i in range(2, len(s)):
            cur_res = 0
            if s[i] == '0':
                if s[i-1] == '1' or s[i-1] == '2':
                    cur_res = prev_prev_res
                else:
                    return 0
            elif s[i-1] == '0':
                cur_res = prev_res
            elif int(s[i-1:i+1]) <= 26:
                cur_res = prev_res + prev_prev_res
            else:
                cur_res = prev_res
            
            prev_prev_res = prev_res
            prev_res = cur_res
            
        return prev_res
            
```
I found this DP solution by self. Well done to me, but I missed many special cases such as '30' '20' '120' '130'. 
 
### [96. Unique Binary Search Trees](https://leetcode.com/problems/unique-binary-search-trees/description/)

```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        res = [0] * (n+1)
        res[0] = 1
        for i in range(1, n+1):
            for j in range(i):
                res[i] += res[j] * res[i-j-1]
        
        return res[n]
             
```
The idea is based on [this](https://leetcode.com/problems/unique-binary-search-trees/discuss/31666/DP-Solution-in-6-lines-with-explanation.-F(i-n)-G(i-1)-G(n-i)). 


## Backtracking

### [301. Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)

```python
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        
        def bt(s, visited, results):
            mi = calc(s)
            if mi == 0:
                return results.append(s)
            ans = []
            for x in range(len(s)):
                if s[x] in ('(', ')'):
                    ns = s[:x] + s[x+1:]
                    if ns not in visited and calc(ns) < mi:
                        visited.add(ns)
                        bt(ns, visited, results)
            return ans
        
        def calc(s):
            a = b = 0
            for c in s:
                a += {'(' : 1, ')' : -1}.get(c, 0)
                b += a < 0
                a = max(a, 0)
            return a + b

        visited = set([s])
        results = []
        bt(s, visited, results)
        return results
```


### [294. Flip Game II](https://leetcode.com/problems/flip-game-ii/description/)

```python
class Solution(object):
    
    _memo = {}
    def canWin(self, s):
        def helper(s):
            memo = self._memo
            if s in memo:
                return memo[s]
            memo[s] = any(s[i:i+2] == '++' and not self.canWin(s[:i] + '--' + s[i+2:]) for i in range(len(s)))
            return memo[s]
        return helper(s)
```
On each step, return I can continue and my opponent cannot continue.
Use memoization to save time by avoiding repeat computation. 
### [37. Sudoku Solver](https://leetcode.com/problems/sudoku-solver/description/)

```python
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        
        solution = None
        self.backtrack(board, collections.defaultdict(set),collections.defaultdict(set),collections.defaultdict(set))
        return solution
    
    def backtrack(self, board, square, hor, ver, solution):
        
        x = y = -1
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    x,y = i,j
        if x == -1: # not finding any empty space
            solution = board
            
        for num in str(range(1, 10)):
            corner_x, corner_y = x // 3, y // 3
            if num not in square[(corner_x,corner_y)] and num not in hor[x] and num not in ver[y]:
                board[x][y] = num
                square[(corner_x,corner_y)].add(num)
                hor[x].add(num)
                ver[y].add(num)
                self.backtrack(board, square, hor, ver, solution)
                if solution:
                    return
                
                board[x][y] = '.'
                square[(corner_x,corner_y)].remove(num) # backtrack
                hor[x].remove(num)
                ver[y].remove(num)
                
        return
        
```
My own backtracking solution. 


### [51. N-Queens](https://leetcode.com/problems/n-queens/description/)

```python
class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        self.backtrack([], [], [], n, res)
        print(res)
        
        
        final_res = []
        for sol in res:
            str_res = [ ['.' for _ in range(n)] for _ in range(n)]
            for i, j in enumerate(sol):
                str_res[i][j] = 'Q'
            final_res.append(map(lambda x: "".join(x), str_res))
                
        return final_res
    
    
    def backtrack(self, queens, xy_sum, xy_diff, n, results):
        '''
        queens:
            its indices are queens horizontal locations
            its values are queens vertical locations
        '''
        
        if len(queens) == n:
            results.append(queens)
            return
        
        for i in range(n):
            _sum = i+len(queens) 
            _diff = i-len(queens) 
            if i not in queens and _sum not in xy_sum and _diff not in xy_diff:
                self.backtrack(queens + [i], xy_sum + [_sum], xy_diff + [_diff], n, results)   
```
PS: Each row and column must have exactly one queen. 

Search row by row. 

Rule: Let `(x, y)` be any queen. For every `(a, b)` , if `a+b == x+y` or `a-b == x-y` or `(a, b)` is in the same row or columns as any queen, `(a, b)` cannot be a queen. 


### [216. Combination Sum III](https://leetcode.com/problems/combination-sum-iii/description/)

```python
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        res = []
        self.backtracking(k, n, 1, [], res)
            
        return res
    
    def backtracking(self, k, n, start, result, results):
    
        if k == 0:
            if n == 0:
                results.append(result)
            return
        
        for x in range(start, 10):
            if n >= x:
                self.backtracking(k-1, n-x, x+1, result + [x], results) # pay attention to "k-1" and "n-x". This is how we reduce the problem. 
            else:
                break
```
Not a hard problem, but I spent too long on it. I need to review it in the future. 

Remeber to use the idea of "problem reduction". We reduce the problem to a smaller problem during backtracking. 

### [211. Add and Search Word - Data structure design](https://leetcode.com/problems/add-and-search-word-data-structure-design/description/)

```python
class WordDictionary:

    def __init__(self):
        self.root = {}
    
    def addWord(self, word):
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node[None] = None

    def search(self, word):
        def find(word, node):
            if not word:
                return None in node
            char, word = word[0], word[1:]
            if char != '.':
                return char in node and find(word, node[char])
            return any(find(word, kid) for kid in node.values() if kid)
        return find(word, self.root)
```
Use prie to store all words. Use recursion for backtracking search.

Remember the usage of `Any()`

### [131. Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/description/)

```python
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        results = []
        self.backtrack(s, [], results)
        return results
    
    def backtrack(self, s, result, results):

        if not s:
            results.append(result)
            return
        for i in range(1, len(s)+1): # pay attention to this len(s)+1. Because we want [:len(s)], so len(s)+1 should be the second parameter of `range`. 
            if s[:i]== s[:i][::-1]:
                self.backtrack(s[i:], result + [s[:i]], results)
        
        
```
In each recursion step, check whether the prefix is a palindrome. If yes, continue to find palindrome in the tail. If not, don't need to search. 

### [90. Subsets II](https://leetcode.com/problems/subsets-ii/description/)

```python
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        _, res = self.rec(nums)
        return res
        
    def rec(self, nums):
        
        if len(nums) == 0:
            return 0, [[]]
        
        res = []
        new_items_count_from_last_step, tail = self.rec(nums[1:])
        
        if len(nums) >= 2 and nums[0] == nums[1]:
            count = 0
            for i in range(new_items_count_from_last_step):
                count += 1
                res.append([nums[0]] + tail[i])

            return count, res + tail
        else:
            count = 0
            for x in tail:
                count += 1
                res.append([nums[0]] + x)
            return count, res + tail
            

```
Idea: If `S[i-1] == S[i]`, we only add S[i] to the newly created items from last step to create the new items, but not all items so far.  

Trick: returning two values. The first value count the number of newly created items for the current step. 


### [79. Word Search](https://leetcode.com/problems/word-search/description/)

```python 
class Solution(object):   
    def exist(self, board, word):
        if not board:
            return False
        for i in xrange(len(board)):
            for j in xrange(len(board[0])):
                if self.dfs(board, i, j, word):
                    return True
        return False

    # check whether can find word, start at (i,j) position    
    def dfs(self, board, i, j, word):
        if len(word) == 0: # all the characters are checked
            return True
        if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
            return False
        tmp = board[i][j]  # first character is found, check the remaining part
        board[i][j] = "#"  # avoid visit agian 
        # check whether can find "word" along one direction
        res = self.dfs(board, i+1, j, word[1:]) or self.dfs(board, i-1, j, word[1:]) \
        or self.dfs(board, i, j+1, word[1:]) or self.dfs(board, i, j-1, word[1:])
        board[i][j] = tmp
        return res
            
```

### [39. Combination Sum](https://leetcode.com/problems/combination-sum/description/)

```python 
class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort() # note this
        res = []
        self.backtracking(candidates, 0, target, [], res)
        return res
        
    def backtracking(self, candidates, start, target, result, results):
        
        if target < 0:
            return 
        elif target == 0:
            results.append(result)
            return
        else:
            for i in range(start, len(candidates)):
                self.backtracking(candidates, i, target-candidates[i], result+[candidates[i]], results)
    
                    
                    
```
Template for backtracking. 

Remember to draw the search tree on paper when using backtracking.

Sorting the candidate is useful in solving problems. 

### [40. Combination Sum II](https://leetcode.com/problems/combination-sum-ii/description/)

```python 
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        candidates.sort()
        self.backtrack(candidates, target, 0, [], res)
        return res
    
    def backtrack(self, nums, target, start, result, results):
        
        if target == 0:
            results.append(result)
        
        for i in range(start, len(nums)):
            
            if i > start and nums[i-1] == nums[i]:
                continue
            
            if nums[i] > target: # cut the search space
                return
            
            self.helper(nums, target-nums[i], i + 1, result + [nums[i]], results)
```
We can cut the search space in backtracking

### [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/)

```python
class Solution(object):
    def letterCombinations2(self, digits): # Iteration
        """
        :type digits: str
        :rtype: List[str]
        """
        mapping = {'2':'abc', 
                  '3':'def',
                  '4':'ghi',
                  '5':'jkl',
                  '6':'mno',
                  '7':'pqrs',
                  '8':'tuv',
                  '9':'wxyz'}
        if len(digits) == 0:
            return []
        
        res = ['']
        for x in digits:
            letters = mapping[x]
            new_res = []
            for r in res:
                for l in letters:
                    new_res += [r + l]
            res = new_res
            
        return res
    def letterCombinations(self, digits): # Recursion
        mapping = {'2':'abc', 
                  '3':'def',
                  '4':'ghi',
                  '5':'jkl',
                  '6':'mno',
                  '7':'pqrs',
                  '8':'tuv',
                  '9':'wxyz'}
        def recurr(digits):
            if len(digits) == 0:
                return []
            if len(digits) == 1:
                return list(mapping[digits[0]])
            letters = mapping[digits[0]]
            res = []
            for l in letters:
                tail = recurr(digits[1:])
                for x in tail:
                    res.append(l+x)
            return res
        return recurr(digits)
    
```



### [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/description/)
```python
class Solution:
    def generateParenthesis(self, n):
        def recur(n):
            if n==1: return set(["()"])
            
            res = set()
            for x in recur(n-1):
                for i in range(len(x)):
                    res.add(x[:i] + "()" + x[i:])
            
            return res
        return list(recur(n))
        
            
```
Recursion solution


## [47. Permutations II](https://leetcode.com/problems/permutations-ii/description/)
```python
class Solution(object):
    def permuteUnique2(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def recur(nums):
            if len(nums) == 1: return [nums]
            dup = set()
            res = []
            for i in range(len(nums)):
                if nums[i] in dup:
                    continue
                dup.add(nums[i])
                for x in recur(nums[:i] + nums[i+1:]):
                    res.append([nums[i]] + x)
            
            return res
            
        return recur(nums)
    
    def permuteUnique(self, nums): # Top-Down Recursion
        
        res = []
        def recur(nums, temp, result):
            if len(nums) == 0:
                result.append(temp)
                return
            dup = set()
            for i in range(len(nums)):
                if nums[i] in dup:
                    continue
                dup.add(nums[i])
                recur(nums[:i] + nums[i+1:],  temp + [nums[i]], result)
           
        recur(nums, [], res)
        return res
```

Recursion solution. Also use TOP-DOWN recursion. 

### [46. Permutations](https://leetcode.com/problems/permutations/description/)

```python
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def per(nums):
            if len(nums) == 1: return [[nums[0]]]
            res = []
            for i,x in enumerate(nums):
                for s in per(nums[:i]+nums[i+1:]):
                    res += [[x] + s]
            return res
        return per(nums)
        
```

### [77. Combinations](https://leetcode.com/problems/combinations/description/)
```python
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
            if len(temp) == k or x > n - k + len(temp) + 1: # cut search space by stop searching if there are not enough numbers
                if not temp:
                    return res
                x = temp.pop() + 1 # General rule: if the return temp condition is not met, just break the condition without inserting node to the temp. Don't insert new node to the temp here. Insert in next iteration
            else:
                temp.append(x) 
                x += 1 
        
        
```
Backtracking model template. 

See comment for cutting search space technique


### [401. Binary Watch](https://leetcode.com/problems/binary-watch/description/)
```python
class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        res = []
        for h in range(0,12):
            for m in range(0, 60):
                if (bin(h) + bin(m)).count('1') == num:
                    res.append('%d:%02d' % (h,m))
        return res
```
Try all different times and return the one with `num` number of `1`.

It is a reverse thinking of solving problem. We work from time to binary, not from binary to time. 

### [78. Subsets](https://leetcode.com/problems/subsets/description/)

```python 
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = [[]]
        for x in nums:
            res += [[x] + i for i in res]
        return res
```     
GOOD PROBLEM. Remember this formula for generating all possible combinations.           

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


## Math

### [50. Pow(x, n)](https://leetcode.com/problems/powx-n/description/)

```python
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n == 0:
            return 1
        if n < 0:
            x = 1/x
            n = -n
        
        return self.myPow(x*x, n/2) if n % 2 == 0 else x * self.myPow(x, n-1)
```

### [60. Permutation Sequence](https://leetcode.com/problems/permutation-sequence/description/)

```python
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        
        return self.permutation(range(1, n+1), k, math.factorial(n))
        
    def permutation(self, nums, k, factorial):
        
        if len(nums) == 1:
            return str(nums[0])
        start_digit = (k-1) // (factorial / len(nums))
        
        return str(nums[start_digit]) + self.permutation(nums[:start_digit] + nums[start_digit+1:],
                                                         1 + (k-1) % (factorial / len(nums)), 
                                                        factorial / len(nums))
        
        
```
This is my own solution. Compute the start digit and do recursion.

### [172. Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/description/)

```python 
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n > 0: 
            n = n // 5
            count += n
           
        return count
```
Count the number of `5` in the factorial

### [357. Count Numbers with Unique Digits](https://leetcode.com/problems/count-numbers-with-unique-digits/description/)

```python
class Solution(object):
    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        n = 10 if n > 10 else n
        
        l = [9, 9, 8, 7, 6 ,5 ,4, 3, 2, 1]
        
        pro = 1
        _sum = 1
        for i in range(n):
            pro *= l[i]
            _sum += pro
            
        return _sum
```
There are 9 options for the first digit, 9 options for the second digit (including 0), 8 optiopns for the third digit. 

### [9. Palindrome Number](https://leetcode.com/problems/palindrome-number/description/)

```python 
    def isPalindrome3(self, x): # Fastest one 
        if x < 0:
            return False

        ranger = 1
        while x / ranger >= 10:
            ranger *= 10

        while x:
            left = x / ranger
            right = x % 10
            if left != right:
                return False
            
            x = (x % ranger) / 10
            ranger /= 100

        return True
```
If the input is 1221. Next step it is 22 (we peeled the head the tail by math calculations). 

### [67. Add Binary](https://leetcode.com/problems/add-binary/description/)

```python
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        
        if len(a) > len(b):
            b = '0' * (len(a) - len(b)) + b
        elif len (a) < len(b):
            a = '0' * (len(b) - len(a)) + a
        
        return self.recur(a,b, '0')
            
        
    def recur(self, a, b, carry):
        
        if len(a) == 0:
            if carry == '1': # remember to add the last carry 
                return '1'
            else:
                return ''
        
        if a[-1] == '0' and b[-1] == '0':
            return self.recur(a[:-1], b[:-1], '0') + carry
        elif a[-1] == '1' and b[-1] == '1':
            return self.recur(a[:-1], b[:-1], '1') + carry
        else:
            if carry =='1':
                return self.recur(a[:-1], b[:-1], '1') + '0'
            else:
                return self.recur(a[:-1], b[:-1], '0') + '1' 
        
```
Use recursion to compute binary addition.

### [168. Excel Sheet Column Title](https://leetcode.com/problems/excel-sheet-column-title/description/)

```
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        
        res = ''
        while n > 0:
            res += chr((n-1) % 26+65)
            n = (n-1) // 26
        
        return res[::-1]
```
Idea: strip the unit digit (26-system) iteratively.

Note this `n-1` rather than `n`



### [628. Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/description/)

```python
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        f = lambda x,y: x*y
        return max(reduce(f, heapq.nlargest(3, nums)), reduce(f, heapq.nsmallest(2, nums) + heapq.nlargest(1, nums)))

```
`heapq.nlargest(n, nums)` is `O(n*nums)`. It means that it can be `O(n)` if n is a constant and nums is `n`.

## Linked List

### General Rules
In most cases, use 'while cur' in the loop condition, but if `cur = cur.next.next...` is possible in the loop, use 'while cur and cur.next'


### [430. Flatten a Multilevel Doubly Linked List](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/description/)

```python
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""
class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return None
        
        
        stack = [head]
        temp_head = Node(0, None, None, None)
        prev = temp_head
        
        while stack:
            node = stack.pop()
            if node.next:
                stack.append(node.next) # we want to search for 'next' later, so we push it into stack earlier. 
            if node.child:
                stack.append(node.child)
            
            
            node.next = None 
            node.prev = prev
            node.child = None
            
            prev.next = node
            prev = node
        
        temp_head.next.prev = None # Make sure the real head does not have a "prev"
        return temp_head.next
        
            
            
```
Depth-first search. 



### [141. Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)
```python 
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        dum = ListNode(0)
        dum.next = head
        slow = dum
        fast = dum.next
        while fast and fast.next:
            if fast is slow: return True
            slow = slow.next
            fast = fast.next.next
        
        return False
```
Use slow-fast pointers approach for linked list cycle problems to avoid additional memory cost

### [142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/description/)
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dum = ListNode(0)
        dum.next = head
        slow = dum
        fast = slow.next
        
        found_cycle = False
        while fast and fast.next:
            if fast is slow:
                found_cycle = True
                break
            fast = fast.next.next
            slow = slow.next
        
        if not found_cycle:
            return None
        
        slow = slow.next 
        while not (slow is dum):
            dum = dum.next
            slow = slow.next
            
        return dum
            
```
See (https://leetcode.com/problems/linked-list-cycle-ii/discuss/44783/Share-my-python-solution-with-detailed-explanation).

This is a Math problem. Drawing the graph makes the solution easier to understand. 

If Slow has moved `k` steps, then Fast has moved `2k+1` steps (it starts at the second node). If they meet at this position, we know `k+nc = 2k + 1` where `c` is the length of the cycle and `n` is an integer and `n>=0`. Then `k = nc-1`. Let `k=a+b` where `a` is the distance between the start of the list to the start of the cycle. Then `a = nc - b - 1` where `nc - b` is the distance between meeting point and the start of the cycle (the distance of the "other side" curve of the cycle). 

### [82. Remove Duplicates from Sorted List II](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dum = ListNode(0)
        dum.next = head
        prev = dum
        while head and head.next: # head and head.next
            if head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                prev.next = head.next
                head = head.next
            else:
                prev = head
                head = head.next
        return dum.next
            

```
use both `head` and `head.next` in the while loop condition if `head` could be assigned by `head.next.next` in the loop.

### [817. Linked List Components](https://leetcode.com/problems/linked-list-components/description/)
```python 
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def numComponents2(self, head, G):
        """
        :type head: ListNode
        :type G: List[int]
        :rtype: int
        """
        g = set(G)
        comp = []
        cur = head
        prev_val = -1
        count = 0
        while cur:
            if cur.val in g and (len(comp) == 0 or comp[-1] == prev_val):
                comp.append(cur.val)
            else:
                if len(comp) > 0: 
                    count += 1
                comp = []
            prev_val = cur.val
            cur = cur.next
        if len(comp) > 0:
            count += 1
            
        
        return count
    
    def numComponents(self, head, G): # better solution
        setG = set(G)
        res = 0
        while head:
            if head.val in setG and (head.next == None or head.next.val not in setG):
                res += 1
            head = head.next
        return res

```
See the better solution. 
Its idea is to count when the condition we cannot build larger component and we don't need to store each component in this problem. 


### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)
```python
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        current_node = head
        previous_node = None
        while current_node:
            next_temp = current_node.next
            current_node.next = previous_node
            previous_node = current_node
            current_node = next_temp
        return previous_node
            
            
```
Template for reversing linked lists. 

## Binary Search

### [287. Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/description/)

```python
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        low = 1
        high = len(nums)
        
        while low < high:
            mid = (low + high) // 2
            count = 0
            for x in nums:
                if x <= mid:
                    count += 1
            if count <= mid:
                low = mid + 1
            else:
                high = mid
        
        return low
```
Find the `mid` (can be any number between 1 and n inclusive). 

If there is no duplicate number, than the count of numbers that are smaller than or equal to `mid` must be `mid`. But this problem defines that there must be at least one duplicate number

Then, if the count of numbers that are smaller than `mid` is smaller than or equal to `mid`, this duplicate number must be greater than `mid`. Otherwise, it is larger than `mid`. This becomes a search problem. 

We use binary search to reduce the search space


### [34. Search for a Range](https://leetcode.com/problems/search-for-a-range/description/)

```python
def searchRange(self, nums, target):
    
    if len(nums) == 0:
        return [-1, -1]
    def findLeftBound(nums, target):
        
        a,b = 0,len(nums)
        
        while a<b:
            m = (a + b) // 2
            if nums[m] >= target:
                b = m
            else:
                a = m + 1
        
        if a >= 0 and a < len(nums) and nums[a] == target:
            return a
        else:
            return -1

    def findRightBound(nums, target):
        
        a,b = 0, len(nums)
        
        while a<b:
            m = (a+b) // 2
            if nums[m] > target:
                b = m
            else:
                a = m + 1
        
        return a
    
    left = findLeftBound(nums, target)
    if left != -1:
        return [left, left + findRightBound(nums[left:], target)-1]
    else:
        return [-1, -1]
```

### [35. Search Insert Position](https://leetcode.com/problems/search-insert-position/description/)

```python
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        a,b = 0, len(nums) 
        while a<=b:
            m = (a + b) // 2
            if nums[m] == target:
                return m
            elif nums[m] > target:
                b = m
            else:
                a = m+1
        
        return a 
```
Template for binary search insertion. Pay attention to `<=`.

The return value is `a` not `m`. 

## Divide and Conquer

### [932. Beautiful Array](https://leetcode.com/problems/beautiful-array/)
```python
class Solution(object):
    def beautifulArray(self, N):
        """
        :type N: int
        :rtype: List[int]
        """
        return self.helper(list(range(1,N+1)))
        
    def helper(self, lst):
        if len(lst)<=2:         
            return lst
        return self.helper(lst[::2]) + self.helper(lst[1::2])
```
Divide and Conquer by odd and even, not by chunk. This is very innovative.

### [241. Different Ways to Add Parentheses](https://leetcode.com/problems/different-ways-to-add-parentheses/description/)
```python
class Solution(object):
    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        if input.isdigit():
            return [int(input)]

        res = []
        for i, x in enumerate(input):
            if x in'+-*':
                a = self.diffWaysToCompute(input[:i])
                b = self.diffWaysToCompute(input[i+1:])
                for x1 in a:
                    for x2 in b:
                        if x == '+':
                            res.append(x1+x2)
                        elif x == '-':
                            res.append(x1-x2)
                        else:
                            res.append(x1*x2)
        return res
    
            
```
Divide and Conquer example.

### [222. Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/description/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        left_height = self.count_height(root.left)
        right_height = self.count_height(root.right)
        
        node_count = 1
        if left_height == right_height:
            node_count += self.count_node_of_complete_tree(left_height) + self.countNodes(root.right)
        else:
            node_count += self.count_node_of_complete_tree(right_height) + self.countNodes(root.left)
        
        return node_count
            
    def count_node_of_complete_tree(self, height):
        
        return sum(pow(2, i) for i in range(height))
            
    def count_height(self, root):
        
        if not root:
            return 0
        
        height = 1
        while root.left:
            root = root.left
            height += 1
        return height
```
If both branches have same lengths, then the last node is on the right branch and the left branch is a complete tree. We only need to run recursion on the right.

If both branches have different lengths, then the last node is on the left branch and the right branch is a complete tree. We only need to run recursion on the left.

### [240. Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/description/)

```python 
    def searchMatrix(self, matrix, target):
        
        top,right = 0, len(matrix[0]) - 1
        top_bound = len(matrix) - 1
        
        while top <= top_bound and right >= 0:
            if matrix[top][right] == target:
                return True
            elif matrix[top][right] > target:
                right -= 1
            else:
                top += 1
        return False 
```
Rule out ineligible rows and columns

## String

### [158. Read N Characters Given Read4 II - Call multiple times
](https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/)

```python

class Solution(object):
    def __init__(self):
        self.queue = []
    
    
    def read(self, buf, n):
        idx = 0
        curr = None
        while curr != 0 :
            buf4 = [""]*4
            l = read4(buf4)
            
            # read 4 chars into the queue
            self.queue.extend(buf4)
            
            # this condition ensures that
            # it stops the loop when the queue is empty (which means all buffer have been read) or 
            # we have read n chars
            curr = min(len(self.queue), n-idx)
            for i in xrange(curr):
                buf[idx] = self.queue.pop(0)
                idx+=1
        return idx
```

### [214. Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/description/)

```python
class Solution(object):
    def shortestPalindrome(self, s):
        A=s+"*"+s[::-1]
        kmp = [0]
        for i in range(1, len(A)):
            previous_i = kmp[i-1]
            
            while(A[previous_i] != A[i] and previous_i > 0 ):
                previous_i = kmp[previous_i-1]
            kmp.append(previous_i+(1 if A[previous_i] == A[i] else 0))
            
        return s[kmp[-1]:][::-1] + s
        
```
KPM based solution

### [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/description/)

```python
class Solution(object):
    def countSubstrings(self, s):
        N = len(s)
        count = 0
        for center in range(2*N - 1):
            left = center // 2
            right = left + center % 2
            
            while left >= 0 and right < N and s[left] == s[right]: # A longer palindrom can only be a palindrom if its inner part is a palindrom. 
                count += 1
                left -= 1
                right += 1
        
        return count
```
Use the idea: A longer palindrom can only be a palindrom if its inner part is a palindrom. 

### [38. Count and Say](https://leetcode.com/problems/count-and-say/description/)

```python
    def countAndSay(self, n):
        
        s = '1'
        for x in range(n-1):
            new_s = []
            for k, g in itertools.groupby(s):
                new_s.append(str(len(list(g))))
                new_s.append(k)
            s = "".join(new_s)
        return s# Definition for an interval.
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
```
Groupby

### [14. Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/description/)

```python
    def longestCommonPrefix(self, strs): # better answer
        common = ''
        for x in zip(*strs):
            _set = set(x)
            if len(_set) == 1:
                common += _set.pop()
            if len(common) == 0: return common
        return common
```
Usage of `zip(*strs)`. 
Example:

input: ["flower","flow","flight"] 

output: [('f', 'f', 'f'), ('l', 'l', 'l'), ('o', 'o', 'i'), ('w', 'w', 'g')]

### [161. One Edit Distance](https://leetcode.com/problems/one-edit-distance/description/)

```python
class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        
        if abs(len(s) - len(t)) > 1:
            return False
        if s == t:
            return False
        s, t = (s, t) if len(s) < len(t) else (t, s)
        for i in range(len(s)):
            if s[i] != t[i]:
                return s[i:] == t[i+1:] or s[i+1:] == t[i+1:] # This is the fast way of comparison
        
        return True
```


### [8. String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/description/)

```python 
    
    def myAtoi(self, str):
        str = str.strip()

        res = re.findall('^[\-\+0]?[0-9]+', str) # know how to use this function. 

        res = int(res[0]) if len(res) > 0 else 0
        if res > 2147483647: return 2147483647
        elif res < -2147483647: return -2147483648
            
        return res
```
Note this Regular Expression findall. 

### [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) == 0:
            return len(s)

        slow,fast = 0,0
        _hash = {}
        max_len = 1
        while fast < len(s):
            if s[fast] in _hash and slow <= _hash[s[fast]]:
                slow = _hash[s[fast]] + 1
            else:
                max_len = max(max_len, fast - slow + 1)
            _hash[s[fast]] = fast
            fast += 1
  
        return max_len
```
`_hash` maps from character to its index. `slow` then uses index of previous repeated character to jump.  

Template: This jumping with index is a template


### [557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/description/)

```python 
class Solution(object):
    def reverseWords2(self, s):
        """
        :type s: str
        :rtype: str
        """
        sl = s.split(' ')
        res = []
        for word in sl:
            res += list(word)[::-1] + [' ']
        
        return "".join(res[:-1])
    
    def reverseWords(self, s): # One line version, faster. 
        return ' '.join(x[::-1] for x in s.split())
```
`join` joins strings with given delimiter. 

### [6. ZigZag Conversion](https://leetcode.com/problems/zigzag-conversion/description/)

```python
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1: return s
        res = ['' for i in range(numRows)]
        movingDown = True
        i = 0
        step = -1
        for x in s:
            res[i] += x
            if i == len(res) - 1 or i == 0:
                step *= -1
                
            if movingDown:
                i += step
            else:
                i -= step
                
        return "".join(res)
```
Use `+=` to append character to a string. It is also fast. 

Use `step` and `step *= -1` to reverse the moving direction. 

### [387. First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/description/)

```python
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        letters='abcdefghijklmnopqrstuvwxyz'
        index = []
        for l in letters:
            if s.count(l) == 1:
                index += [s.index(l)]
        return min(index) if len(index) > 0 else -1
```
Use `count`

## Sort

### [148. Sort List](https://leetcode.com/problems/sort-list/description/)

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        return self.merge_sort(head)
    
    def merge_sort(self, head):
        
        if not head:
            return None
        
        if not head.next:
            return head
        
        dummy = ListNode(0)
        slow, fast, slow_prev = head, head, dummy
        
        # partition
        while fast and fast.next:
            fast = fast.next.next
            slow_prev = slow
            slow = slow.next
        slow_prev.next = None
        
        cur1 = self.merge_sort(head)
        cur2 = self.merge_sort(slow)
        
        dummy = ListNode(0)
        cur = dummy
        
        while cur1 and cur2:
            if cur1.val < cur2.val:
                cur.next = cur1
                cur1 = cur1.next
            else:
                cur.next = cur2
                cur2 = cur2.next
            cur = cur.next
        
        if cur1:
            cur.next = cur1
        
        if cur2:
            cur.next = cur2
            
        return dummy.next
```
Merge sort linked list



## String

### [242. Valid Anagram](https://leetcode.com/problems/valid-anagram/description/)
```python
class Solution(object):
    def isAnagram2(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        
        return collections.Counter(s) == collections.Counter(t)
    
    def isAnagram(self, s, t):
            return all([s.count(c)==t.count(c) for c in string.ascii_lowercase])
```
Example usage of `all`. 

It is also normal to see that we use alphabet to search and check chars in strings. 

## Tree

### [285. Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        
        succ = None
        
        while root:
            if p.val < root.val:
                succ = root
                root = root.left
            else:
                root = root.right 
        
        return succ
```

### [979. Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distributeCoins(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        """
        
        """
        res = [0]
        def rec(root, res):
            
            if not root: return 0
            left = rec(root.left, res)
            right = rec(root.right, res)
            res[0] += abs(left) + abs(right)
            
            return root.val + left + right - 1
                
            
        rec(root, res)
        return res[0]

```
This is a very good and typical problem. We count the number of coins move through each edges in the tree. 

### [863. All Nodes Distance K in Binary Tree](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/description/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        graph = collections.defaultdict(list)
        def connect(parent, child): # build a graph. Remember this function
            if parent and child:
                graph[parent.val].append(child.val)
                graph[child.val].append(parent.val)
            if child.left:
                connect(child, child.left)
            if child.right:
                connect(child, child.right)
            
        connect(None, root)
        level = [target.val]
        visited = set(level)
        for i in range(K):
            level = [y for x in level for y in graph[x] if y not in visited] # Remember this way of BFS 
            visited |= set(level) 
        
        return level
        
        
```
We firstly build a graph from the tree. Then perform the BFS K times to find the nodes.
V

### [671. Second Minimum Node In a Binary Tree](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/description/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root or not root.left or not root.right:
            return -1
        
        smallest = root.val
        res = [float('inf')]
        def traverse(tree):
            if smallest < tree.val < res[0]:
                res[0] = tree.val
            if tree.left:
                traverse(tree.left)
            if tree.right:
                traverse(tree.right)
        traverse(root.left)
        traverse(root.right)
        return res[0] if res[0] != float('inf') else -1
```
Travese each node and get the value that is only greater than the smallest value (root.val). 

### [116. Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

```python 

# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        
        if not root:
            return
        
        cur = root
        next = root.left
        
        while cur.left:
            cur.left.next = cur.right
            if cur.next:
                cur.right.next = cur.next.left
                cur = cur.next
            else:
                cur = next
                next = cur.left
        
```
Treat previous level as a linked list and iterate through this linked list. In each iteration, connect children of nodes in the linked list


### [117. Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/)

```python
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        
        cur = root
        _next = None
        prev = None
        
        while cur:# level 1
            
            if prev:
                if cur.left and cur.right:
                    prev.next = cur.left
                    cur.left.next = cur.right
                    prev = cur.right
                elif cur.left:
                    prev.next = cur.left
                    prev = prev.next
                elif cur.right:
                    prev.next = cur.right
                    prev = prev.next
            else:
                if cur.left and cur.right:
                    _next = cur.left
                    cur.left.next = cur.right
                    prev = cur.right
                elif cur.left:
                    _next = cur.left
                    prev = cur.left
                elif cur.right:
                    _next = cur.right
                    prev = cur.right

            if cur.next:
                cur = cur.next
            else:
                cur = _next
                _next = None
                prev = None
```
(I forgot the case where a 'cur' has two children.) 

### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/)

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        
        if not root:
            return None 
        
        if root.val == p.val or root.val == q.val:
            return root
        
        a = self.lowestCommonAncestor(root.left, p, q)
        b = self.lowestCommonAncestor(root.right, p, q)
        if a and b:
            return root
        elif a:
            return a
        else:
            return b
        
        

```
If both p and q are in left, then the LCA is in left.

If p and q are split in left and right, then this node is the LCT.


### [235. Lowest Common Ancestor of a Binary Search Tree]()

```python
    
    def lowestCommonAncestor(self, root, p, q):
        
        while root:
            if p.val < root.val > q.val:
                root = root.left
            elif p.val > root.val < q.val:
                root = root.right
            else:
                return root
        
```
Use the property of the binary search tree. If a's value is between c and d, then a must be the lowest common ancester of c and d. 


### [106. Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)


```python
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not postorder or not inorder:
            return None
    
        root = TreeNode(postorder.pop())
        inorder_index = inorder.index(root.val)
        
        root.right = self.buildTree(inorder[inorder_index+1:], postorder) # We don't need to slice the postorder. This saves time
        root.left = self.buildTree(inorder[:inorder_index], postorder)
        
        
        return root
```
We don't need to slice the postorder. This saves time

### [652. Find Duplicate Subtrees](https://leetcode.com/problems/find-duplicate-subtrees/description/)
```python 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    def findDuplicateSubtrees(self, root): 
        
        def hash_tree(root):
            if not root:
                return None
            tree_id = (hash_tree(root.left), root.val, hash_tree(root.right))
            count[tree_id].append(root)
            return tree_id
        
        count = collections.defaultdict(list)
        hash_tree(root)
        return [nodes.pop() for nodes in count.values() if len(nodes) >= 2]
        
        
            
```
This is how do we hash a tree.
### [449. Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/description/)
```python 
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return ''
        stack = [root]
        res = []
        while stack:
            x = stack.pop()
            res.append(str(x.val))
            
            if x.right:
                stack.append(x.right)
            if x.left:
                stack.append(x.left)
        
        return ' '.join(res)
                
            
            

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        queue = collections.deque(int(s) for s in data.split())
        def build(min_val, max_val):
            if queue and min_val < queue[0] < max_val:
                root = TreeNode(queue.popleft())
                root.left = build(min_val, root.val)
                root.right = build(root.val, max_val)
                return root
            return None
        return build(float('-inf'), float('inf'))
                
```
See function `build`. It is an O(n) way to build a binary search tree given a string from pre-ordered tree traversal. 



### [230. Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        _, res = self.helper(root, k)
        
        return res.val
        
        
    def helper(self, root, k):
        
        if not root:
            return 0, None
        
        count_left, kthSmallest_node = self.helper(root.left, k)
        
        if kthSmallest_node:
            return -1, kthSmallest_node
        
        if count_left == k - 1:
            return -1, root
        
        count_right, kthSmallest_node = self.helper(root.right, k - count_left - 1)
        if kthSmallest_node:
            return -1, kthSmallest_node
        
        return count_left + count_right + 1, None
```
This is my own solution. 

We first search kth smallest element in the left. If we find it, return it and that's it. If we cannot find it, search (k-1-left_nodes_count)th element in the right ("1" denotes the count of the root node). If found, return it and that's it. If we still cannot find it, return the total count of nodes and let the parent continue to do the search. 

### [144. Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)

```python 
    def preorderTraversal(self, root):
        
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            x = stack.pop()
            
            res.append(x.val)

            if x.right:
                stack.append(x.right)
            if x.left:
                stack.append(x.left)
        return res
```
Use stack to do the preorder traversal. 

### [101. Symmetric Tree](https://leetcode.com/problems/symmetric-tree/description/)

```python 
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def sym(a,b):
            if a is None and b is None:
                return True
            elif a is None or b is None:
                return False
            
            return a.val == b.val and sym(a.right, b.left) and sym(a.left, b.right)
        
        if not root:
            return True
        else:
            return sym(root.left, root.right)
        
```

### [95. Unique Binary Search Trees II](https://leetcode.com/problems/unique-binary-search-trees-ii/description/)

```python 
class Solution(object):
    def generateTrees(self, n):
        """
        :type n: int
        :rtype: List[TreeNode]
        """
        def generate(values):
            if len(values) == 1:
                return [TreeNode(values[0])]
            if len(values) == 0:
                return []
            
            res = []
            for i, x in enumerate(values):
                left_set = generate(values[:i])
                right_set = generate(values[i+1:])
                
                if len(left_set) == 0: # 
                    for r in right_set:
                        node = TreeNode(x)
                        node.right = r
                        res.append(node)
                        
                elif len(right_set) == 0:
                    for l in left_set:
                        node = TreeNode(x)
                        node.left = l
                        res.append(node)                
                else:
                    for l in left_set: 
                        for r in right_set:
                            node = TreeNode(x)
                            node.left = l
                            node.right = r
                            res.append(node)
            
            return res
        
        return generate(range(1,n+1))
```
Example of how to take cartesian product of two lists (`left_set` and `right_set`)

## Heap

### [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/description/) (Facebook)

```python
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
```
Use heap to maintain the ending time of happening meetings.


### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

```python
class Solution(object):

    def findKthLargest(self, nums, k):
        return heapq.nlargest(k, nums)[k-1]
```
Use `heapq`

## Depth First Search

## [114. Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/)
```python
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child

class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return None
        c
        
        stack = [head]
        temp_head = Node(0, None, None, None)
        prev = temp_head
        
        while stack:
            node = stack.pop()
            if node.next:
                stack.append(node.next)
            if node.child:
                stack.append(node.child)
            
            
            node.next = None
            node.prev = prev
            node.child = None
            
            prev.next = node
            prev = node
        
        temp_head.next.prev = None # Make sure the real head does not have a "prev"
        return temp_head.next
 ```   


### [547. Friend Circles](https://leetcode.com/problems/friend-circles/description/)

```python

class Solution(object):
    def findCircleNum2(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        visited = set()
        circle_count = 0
        for i in range(len(M)):
            if i not in visited:
                stack = [i]
                visited.add(i)
                while stack:
                    x = stack.pop()
                    for j in range(len(M[x])):
                        if M[x][j] == 1 and j not in visited:
                            stack.append(j)
                            visited.add(j)
                circle_count += 1
                
        return circle_count
    
    def findCircleNum(self, M): # recursive DFS
        
        visited = set()
        def rec(root):
            for i in range(len(M[root])):
                if M[root][i] and i not in visited:
                    visited.add(i)
                    rec(i)
                    
        count = 0
        
        for i in range(len(M)):
            if i not in visited:
                visited.add(i)
                count +=1
                rec(i)
        return count
                                             
```
Two way of depth first search.
The depth frist search might not be graphically comprehensible. It can also be applied with abstract rules (like friendship in this case)


### [261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/description/)

```python
class Solution(object):
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        if len(edges) == 0:
            if n == 1:
                return True
            else: 
                return False
        
        neighbors = collections.defaultdict(list)
        
        for x, y in edges:
            neighbors[x].append(y)
            neighbors[y].append(x)
        
        visited = [0 for _ in range(n)]
        stack = [[-1, edges[0][0]]]
        visited[edges[0][0]] = -1
        node_count = 0
        
        while stack:
            parent, x = stack.pop()
            node_count += 1

            no_unvisited_neighbor = True
            for nei in neighbors[x]:
                if visited[nei] == 0:
                    stack.append([x, nei])
                    visited[nei] = -1
                    no_unvisited_neighbor = False
                elif nei != parent and visited[nei] == -1:
                    return False
            if no_unvisited_neighbor:
                visited[x] = 1
        print(node_count)
        
        if node_count < n:
            return False

        return True
```
Use DFS to detect cycle. 

## Graph

### [399. Evaluate Division](https://leetcode.com/problems/evaluate-division/description/)

```python
class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """

        # build the graph

        # graph: {in: out}

        graph = collections.defaultdict(set)
        for i in range(len(equations)):
            graph[equations[i][0]].add((equations[i][1], values[i]))
            graph[equations[i][1]].add((equations[i][0], 1 / float(values[i])))

        res = []
        for i, (a, b) in enumerate(queries):
            # ["a", "b"]
            if a not in graph or b not in graph:
                res.append(-1.0)
                continue
            stack = [(a, 1.0)]
            visit = set([a])
            while stack:
                x, cumprod = stack.pop()
                if x == b:
                    res.append(cumprod)
                for out, val in graph[x]:
                    if out not in visit:
                        visit.add(out)
                        stack.append((out, cumprod * val))
            if len(res) < i + 1:
                res.append(-1.0)
        return res
```
Turn the computation into a graph. Each node is a letter and each edge `(a,b)`'s weight is the value when of `a/b`. 

### [310. Minimum Height Trees](https://leetcode.com/problems/minimum-height-trees/description/)

```python
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        if n == 1:
            return [0]
        neighbours = collections.defaultdict(list)
        degrees = collections.defaultdict(int) # we will use degree to identify the leaf. Leaf with have degree = 1. 
        for u,v in edges:
            neighbours[u].append(v)
            neighbours[v].append(u)
            degrees[u] += 1 
            degrees[v] += 1
        
        # find leaves
        prelevel = []
        for u in degrees:
            if degrees[u] == 1:
                prelevel.append(u)
        
        visited = set(prelevel)
        i = 0
        while len(visited) < n:
            thislevel = []
            for u in prelevel:
                for nei in neighbours[u]:
                    if nei not in visited:
                        degrees[nei] -= 1
                        if degrees[nei] == 1:
                            thislevel.append(nei)
                            visited.add(nei)
            prelevel = thislevel
            i += 1
        
        return prelevel
                
```

The minimum height tree is the centre of the graph, which is the center of the longest path in the tree.

THe solution is to gradually remove the leaves until hit the center. 

### [332. Reconstruct Itinerary](https://leetcode.com/problems/reconstruct-itinerary/description/)

```python 
class Solution(object):

    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        tickets_dict = collections.defaultdict(list)
        for x, y in tickets:
            tickets_dict[x].append([False, y])  # [is_used, destination]
        for x in tickets_dict:
            tickets_dict[x].sort(key=lambda x: x[1])

        results = []

        self.backtrack('JFK', tickets_dict, ['JFK'], results, len(tickets))

        return results[0]


    def backtrack(self, start, tickets, result, results, tickets_num):

        if len(result) == tickets_num + 1:
            results.append(result)
            return

        for ticket in tickets[start]:
            if ticket[0]:  # if the ticket has been used
                continue
            ticket[0] = True
            self.backtrack(ticket[1], tickets, result + [ticket[1]], results, tickets_num)
            if len(results) > 0:
                return

            ticket[0] = False  # backtrack, reset the ticket
        return


```
My own backtracking answer. 


### [207. Course Schedule](https://leetcode.com/problems/course-schedule/description/)

```python
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
    
        graph = [[] for _ in xrange(numCourses)] 
        visited = [0 for _ in xrange(numCourses)] # record visits
        
        for x,y in prerequisites:
            graph[x].append(y)
        
        def dfs(i, visited):
            if visited[i] == -1: # when this node has not been traced back
                return False
            if visited[i] == 1: # when this node has been traced back
                return True
            
            visited[i] = -1
            for out in graph[i]:
                if not dfs(out, visited):
                    return False
            visited[i] = 1
            return True
        
        for i in xrange(numCourses):
            if not dfs(i, visited):
                return False
        
        return True
            
            
            
```
Template.

This is a very important and good question. 

The idea is to find a cycle in the directed graph. 

Remember how do we use an array of numbers to record node visit. There are three states for each node (unvisited: 0, visiting: -1, visited: 1). In DFS, the difference between visited and visiting is whether we have "traced back". 



### [210. Course Schedule II](https://leetcode.com/problems/course-schedule-ii/description/)

```python
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        
        in_degrees = [0 for _ in xrange(numCourses)]
        neighbours = [[] for _ in xrange(numCourses)]
        
        for x,y in prerequisites:
            neighbours[y].append(x)
            in_degrees[x] += 1 # courses without prerequisite are those with in-degrees as 0. 
        
        zero_in_degree_nodes = [] # This contains without prerequisite  
        
        for i, in_degree in enumerate(in_degrees):
            if in_degree == 0:
                zero_in_degree_nodes.append(i)
        
        res = []
        count = 0 
        while zero_in_degree_nodes: 
            x = zero_in_degree_nodes.pop()
            res.append(x)
            
            for neighbour in neighbours[x]:
                in_degrees[neighbour] -= 1
                if in_degrees[neighbour] == 0:
                    zero_in_degree_nodes.append(neighbour)
            count += 1 # count the number of steps
            
        if count == numCourses: # if count of 'removing' if less than the total number of nodes, then there must be a cycle. 
            return res
        else: # we return [] when there is a cycle. 
            return []
```
Gradually remove courses with 'in-degree=0' until all coursea are removed. 

## Greedy

### [253. Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/description/)

```python
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
        if len(intervals) == 0:
            return 0
        
        starts = []
        ends = []
        for i in intervals:
            starts.append(i.start)
            ends.append(i.end)
        
        starts.sort()
        ends.sort()

        num_rooms = available = 0
        
        s = e = 0
        
        while s <len(starts):
            if starts[s] < ends[e]:
                if available == 0:
                    num_rooms += 1
                else:
                    available -= 1
                s += 1
            else:
                available += 1
                e += 1
                
        
        return num_rooms
```
Sort both starts and ends. 
Iterate start and end one at a time. 
Think it in a real-world senario. 

### [881. Boats to Save People](https://leetcode.com/problems/boats-to-save-people/description/)

```python
class Solution:
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        people.sort()
        
        boat = 0
        
        i, j = 0, len(people) - 1
        
        while i <= j:
            if people[i] + people[j] <= limit:
                i += 1
                j -= 1
            else:
                j -= 1
            boat += 1
            
        return boat                
```
Use two pointer and greedy.
The greedy idea is to try to put a heavy people in a boat with a light people to fill the gap. This will save the space optimally. 

### [135. Candy](https://leetcode.com/problems/candy/description/)

```python
class Solution:
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        res = [1 for i in range(len(ratings))]
        
        for i in range(1, len(ratings)):
            if ratings[i-1] < ratings[i]:
                res[i] = res[i-1] + 1
        
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i+1] and res[i] <= res[i+1]:
                res[i] = res[i+1] + 1       
        
        
        return sum(res)
                
```
Two-way scan.

### [763. Partition Labels](https://leetcode.com/problems/partition-labels/description/)

```python
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        sizes = []
        
        while S:
            
            i = 1
            while set(S[:i]) & set(S[i:]):
                i += 1
            sizes.append(i)
            S = S[i:]
        return sizes

    def partitionLabels(self, S): # solution 2. Much faster
        d = {c:i for i,c in enumerate(S)}
        
        ans = []
        i = pre = 0
        while i < len(S):
            end = d[S[i]]
            while i <= end:
                end = max(end, d[S[i]])
                i += 1
            ans.append(i-pre)
            pre = i
        
        return ans
```
Remember the use of `set` and `&`. 

The second solution records the last index of each characeter. For each element in the string, if this element is before its last appearance, it should be in the same partition as its last appearance. Then we expand the current partition's end with the this element's last appearance's index. When we hit an 'end', we hit the end of the partition. 

### [406. Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/description/)

```python
class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda (x,y):(-x, y))
        
        res = []
        for p in people:
            res.insert(p[1], p)
        
        return res
```
See [here](https://leetcode.com/problems/queue-reconstruction-by-height/discuss/89345/Easy-concept-with-PythonC++Java-Solution)

### [621. Task Scheduler](https://leetcode.com/problems/task-scheduler/description/)

```python
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
```

[Explanation](https://leetcode.com/problems/task-scheduler/discuss/104500/Java-O(n)-time-O(1)-space-1-pass-no-sorting-solution-with-detailed-explanation)

### [45. Jump Game II](https://leetcode.com/problems/jump-game-ii/description/)
```python
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        last_max_reach_point, current_max_reach_point = 0, 0
        step = 0
        for i in range(len(nums)-1):
            current_max_reach_point = max(i + nums[i], current_max_reach_point)
            if i == last_max_reach_point:
                last_max_reach_point = current_max_reach_point
                step += 1
        
        return step
```

Hard greedy problem

Strategy: 
For every position `i`, we find the max possible reach before and including `i` . Later, if we encounter this max possible reach, we increment a step. Then we have a new max possible reach. 

### [55. Jump Game](https://leetcode.com/problems/jump-game/description/)

```python
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        furthest_destination = 0
        for i,n in enumerate(nums):
            
            if i > furthest_destination:
                return False
            furthest_destination = max(furthest_destination, i+n)
                
        return True
```
Compute the 'furthest_destination' on each node. 

### [659. Split Array into Consecutive Subsequences](https://leetcode.com/problems/split-array-into-consecutive-subsequences/description/)

```python
class Solution(object):
    def isPossible(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """

        left = collections.Counter(nums) # record how many 'i' do we left in nums. 
        end = collections.Counter() # end[i] is the count of consecutive sequences ending at i. 
        
        for i in nums:
            if not left[i]:
                continue
            
            left[i] -= 1
            
            if end[i-1]:
                end[i-1] -= 1
                end[i] += 1
            elif left[i+1] and left[i+2]:
                left[i+1] -= 1
                left[i+2] -= 1
                
                end[i+2] += 1
            else:
                return False
            
        return True
                
        
```
Idea: we build the longest consecutive with length greater 3 by using numbers of `nums`. If we cannot build anymore, we return False.




## Bit Manipulation

### [477. Total Hamming Distance](https://leetcode.com/problems/total-hamming-distance/description/)Facebook

```python
class Solution(object):
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum((_tuple.count('0') * _tuple.count('1')) for _tuple in zip(*map(lambda x: '{0:032b}'.format(x), nums)))
        
```


### [136. Single Number](https://leetcode.com/problems/single-number/description/)

```python
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return reduce(operator.xor, nums)
```
Example of XOR.

## Breadth First Search

### [364. Nested List Weight Sum II](https://leetcode.com/problems/nested-list-weight-sum-ii/description/)

```python
class Solution(object):
    def depthSumInverse(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        cur_level = nestedList
        _sums = []
        while cur_level:
            cur_sum = 0
            next_level = []
            for e in cur_level:
                if e.isInteger():
                    cur_sum += e.getInteger()
                else:
                    next_level.extend(e.getList())
            
            _sums.append(cur_sum)
            cur_level = next_level
            
        res = 0
        fac = 1
        while _sums:
            res = res + fac*_sums.pop()
            fac += 1
        
        return res 

```
Use the stack `_sum` to record the sum of each level and pop them inversely. 

### [127. Word Ladder](https://leetcode.com/problems/word-ladder/description/)

```python
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        queue = collections.deque([[beginWord, 1]])
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz': # use this approach to find next item
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, length + 1])
        return 0
```

## Minmax

### (464. Can I Win)[https://leetcode.com/problems/can-i-win/description/]

```python
class Solution(object):
    
    memo = {}
    def canIWin(self, maxChoosableInteger, desiredTotal):
        """
        :type maxChoosableInteger: int
        :type desiredTotal: int
        :rtype: bool
        """
        if (1 + maxChoosableInteger) * maxChoosableInteger/2 < desiredTotal:
            return False
        
        return self.helper(range(1, maxChoosableInteger + 1), desiredTotal)

        
    def helper(self, nums, desiredTotal):
        
        _hash = str(nums)
        if _hash in self.memo:
            return self.memo[_hash]
        
        if nums[-1] >= desiredTotal:
            self.memo[_hash] = True
            return True
        
        for i in range(len(nums)):
            if not self.helper(nums[:i] + nums[i+1:], desiredTotal - nums[i]):
                self.memo[_hash] = True
                return True
        
        self.memo[_hash] = False
        return False
```

## Stack

### [496. Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/description/)

```python 
class Solution:
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        stack = []
        sols = {}
        
        for x in nums2:
            while len(stack) > 0 and stack[-1] < x:
                sols[stack.pop()] = x
            stack.append(x)
        
        return [sols.get(x, -1) for x in nums1]
```
Very good stack problem.

Use '[9, 8, 7, 3, 2, 1, 6]' as an input. 
We insert the top descreasing array into the stack. Then the stack will be `[9, 8, 7, 3, 2, 1]` and we pop all elements that are smaller than `6`. All poped elements `[3,2,1]` are those smaller than `6` and sit on the left of `6`. 


### [556. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/description/)

```python
class Solution:
    def nextGreaterElements(self, nums):
        nums = nums
        stack, res = [], [-1] * len(nums)
        for i in range(len(nums)):
            while stack and (nums[stack[-1]] < nums[i]):
                res[stack.pop()] = nums[i]
            stack.append(i)
        
        for i in range(len(nums)):
            while stack and (nums[stack[-1]] < nums[i]):
                res[stack.pop()] = nums[i]
            stack.append(i)
            
        return res
```
The input is a circular array. 
A general idea of handling circular array is to double the input array and do things on it.

A better idea here is to run the algorithm twice so we don't need to copy paste the array. In the second run, our stack will contain from the tail of the input and the algorithm can continue and behave as working in a circular array.


## Linkedin

### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/description/)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        h = [(node.val, node) for node in lists if node]
        heapq.heapify(h)
        res = cur = ListNode(0)
        while h: 
            _, node = h[0]
            cur.next = node
            if node.next:
                heapq.heapreplace(h, (node.next.val, node.next))
            else:
                heapq.heappop(h)
            cur = cur.next
        
        return res.next
                
```
Not work in python 3. Use PriorityQueue() instead.


### [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/) 
```python
import collections


class Solution:
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """

        """
        S = "ADOBECODEBANC", T = "ABC"

        """
        need_count = len(t)
        need = collections.Counter(t)

        i = 0
        # use a sliding window to scan the string
        shortest_len = len(s)
        start = 0
        end = len(s) - 1
        for j in range(len(s)):
            # expand the window until all letters in t are covered.
            if s[j] in need:
                if need_count > 0 and need[s[j]] > 0:
                    need_count -= 1
                need[s[j]] -= 1

                # if the window contain more than enough characters

                while i <= j and (s[i] not in need or need[s[i]] < 0):
                    if s[i] in need:
                        need[s[i]] += 1
                    i += 1
                    # if the sequence is smaller than min_seq, record it
                new_len = j - i + 1

                if need_count == 0 and new_len < shortest_len:
                    start = i
                    end = j
                    shortest_len = new_len

        if need_count > 0:
            return ""

        return s[start:end + 1]


s = Solution()
print(s.minWindow("ADOBECODEBANC", "ABC"))

```

## Design
### [297. Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
```python
def serialize(self, root):
    preorder = ''
    if not root:
        preorder += ',None'
        return preorder
    preorder += ','+str(root.val)
    preorder += self.serialize(root.left)
    preorder += self.serialize(root.right)
    return preorder

def deserialize(self, encode_data):
    pos = -1
    data = encode_data[1:].split(',')
    for i in xrange(len(data)):
        if data[i] == 'None':
            data[i] = None
        else:
            data[i] = int(data[i])
    root, count = self.buildTree(data, pos)
    return root
    
def buildTree(self, data, pos):
    pos += 1
    if pos >= len(data) or data[pos]==None:
        return None, pos
        
    root = TreeNode(data[pos])
    root.left, pos = self.buildTree(data, pos)
    root.right, pos = self.buildTree(data, pos)
    return root, pos
```
https://leetcode.com/problems/serialize-and-deserialize-binary-tree/discuss/74434/Python-preorder-recursive-traversal