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
            if carry == '1':
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
        