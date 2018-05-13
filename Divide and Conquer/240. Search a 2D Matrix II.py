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