class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for x in s:
            if x == '[' or x == '{' or x == '(':
                stack.append(x)
            else:
                if len(stack) == 0 or x == ']' and stack.pop() != '[' or x == '}' and stack.pop() != '{' or x == ')' and stack.pop() != '(':
                    return False

        return len(stack) == 0
                