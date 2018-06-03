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