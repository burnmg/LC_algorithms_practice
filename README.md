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
        
            
            # Ruolin's Leetcode Practice NoteBook
A repo for Ruolin's LC practice

## Array

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
        res = [1]
        
        for _ in range(rowIndex):
            res = [x+y for x,y in zip([0]+res, res+[0])]
        
        return res
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
The idea is based on [this](https://leetcode.com/problems/unique-binary-search-trees/discuss/31666/DP-Solution-in-6-lines-with-explanation.-F(i-n)-G(i-1)-*-G(n-i)). 



## Backtracking

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


[47. Permutations II](https://leetcode.com/problems/permutations-ii/description/)
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
        return s
```
Groupby

### [14. Longest Common Prefix]()

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
Usage of `zip(*strs)'. 

Example:

input: ["flower","flow","flight"] 

output: [('f', 'f', 'f'), ('l', 'l', 'l'), ('o', 'o', 'i'), ('w', 'w', 'g')]


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

### [215. Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

```python
class Solution(object):

    def findKthLargest(self, nums, k):
        return heapq.nlargest(k, nums)[k-1]
```
Use `heapq`

## Depth First Search

### []()

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


## Graph

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

### []()
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