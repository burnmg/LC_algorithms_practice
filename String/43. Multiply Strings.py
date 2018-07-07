class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        mul_table = {}
        for i in range(10):
            for j in range(10):
                mul_table[(str(i), str(j))] = i * j
        
        return str(self.rec_mul(num1, num2, mul_table))
        
    def rec_mul(self, num1, num2, mul_table):
        
        if len(num1) == 1 and len(num2) == 1:
            return mul_table[(num1, num2)]
        elif len(num1) == 1:
            factor = 1
            res = 0
            for i2 in range(len(num2)-1, -1, -1):
                res += mul_table[(num1, num2[i2])]  * factor
                factor *= 10 
            return res
        elif len(num2) == 1:
            factor = 1
            res = 0
            for i1 in range(len(num1)-1, -1, -1):
                res += mul_table[(num1[i1], num2)]  * factor
                factor *= 10 
            return res
        
        factor = 1
        res = 0
        for i1 in range(len(num1)-1, -1, -1):
            res += self.rec_mul(num1[i1], num2, mul_table) * factor
            factor *= 10
        return res
            
    def multiply2(self, num1, num2): # without recursion 
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """

        mul_table = {}
        for i in range(10):
            for j in range(10):
                mul_table[(str(i), str(j))] = i * j
        
        res = 0
        factor_i = 1
        for i in reversed(num1):
            factor_j = 1
            for j in reversed(num2):
                res += mul_table[(str(i), str(j))] * factor_i * factor_j
                factor_j *= 10
            factor_i *= 10
        
        return str(res)

            