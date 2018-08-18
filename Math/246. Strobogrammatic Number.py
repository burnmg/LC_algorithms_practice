class Solution(object):
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        num = str(num)
        if len(num) == 0:
            return True
        
        i, j = 0, len(num) - 1
        
        while i <= j:
            if num[i] == '6' and num[j] == '9' or num[i] == '8' and num[i] == '8' or num[i] == '9' and num[j] == '6' or num[i] == '1' and num[j] == '1' or num[i] == '0' and num[j] == '0':
                i += 1
                j -= 1
            else:
                return False

            
        return True