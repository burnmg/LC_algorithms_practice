class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if len(matrix) == 0:
            return []
        top,down,left,right = 0, len(matrix)-1, 0, len(matrix[0])-1

        res = []
        while top <= down and left <= right:
            if top == down:
                for i in range(left, right+1):
                    res.append(matrix[top][i])         
            elif left == right:
                for i in range(top, down+1):
                    res.append(matrix[i][left])
            else:
                
                for i in range(left, right):
                    res.append(matrix[top][i])
                for i in range(top, down):
                    res.append(matrix[i][right])
                for i in range(right, left, -1):
                    res.append(matrix[down][i])                  
                for i in range(down, top, -1):
                    res.append(matrix[i][left])
            
                
            top += 1
            down -= 1
            left += 1
            right -= 1
        
        return res