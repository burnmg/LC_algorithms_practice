class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        d = {
            1:'I',
            4:'IV',
            5:'V',
            9:'IX',
            10:'X',
            40:'XL',
            50:'L',
            90:'XC',
            100:'C',
            400:'CD',
            500:'D',
            900:'CM',
            1000:'M'
            
        }
        l = [1,4,5,9,10,40,50,90,100,400,500,900,1000]
        res = ''
        while num != 0:
            divisor = l[bisect.bisect_right(l, num) - 1]
            digit_count = num // divisor
            res += digit_count * d[divisor]
            num = num % divisor
            
        return res
            
            
    