class Solution(object):
    def myAtoi2(self, str):
        """
        :type str: str
        :rtype: int
        """
        numbers = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'])
        res = 0
        encountered_number = False
        encountered_sign = False
        is_negative = False
        str = str.strip()
        for x in str:
            if not encountered_number:
                if x in numbers:
                    encountered_number = True
                    res = int(x)
                    continue
                
                if encountered_sign and x not in numbers:
                    return 0
                elif x == '-':
                    is_negative = True
                    encountered_sign = True
                elif x == '+':
                    encountered_sign = True                    
                else:
                    return 0 
            else:
                if x in numbers:
                    res = res*10 + int(x)
                    if res >= 2147483648: return -2147483648 if is_negative else 2147483647
                else:
                    return -1*res if is_negative else res
        return -1*res if is_negative else res
    
    def myAtoi(self, str):
        str = str.strip()
        res = re.findall('^[\-\+0]?[0-9]+', str)
        res = int(res[0]) if len(res) > 0 else 0
        if res > 2147483647: return 2147483647
        elif res < -2147483647: return -2147483648
            
        return res
            
                