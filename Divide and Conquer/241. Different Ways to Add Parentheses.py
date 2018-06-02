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
    
            
        