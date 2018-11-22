class Solution:
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        # no greater time today, we should return the smallest time 
        # 19:34

        # Reuse 1,9,3, to make a valid and smallest next time compared to 19:34

        # 19:34 we want to replace 4 with the next largest number

        """
        19:39 / 1,3,9

        HH:MM. 

        ### Rule
         00 <=HH < 23

         00 <= HH < 59
         ###

         given a two digits number, a set of avaiable digit. Find the smallest two digits that are grteater than this current two digits. The selected number must satisfy the rule. 

         If cannot find a setting for MM, we try to find one for HH, and set MM to the smallest

         If we cannot find the settings for both. I will find the smallest setting for both HH and MM

        """

        """
        19:43 / 1,3, 4, 9
        """

        # sort the avabale digits
        hh, mm = time.split(":")
        ava_digits = list(set([int(digit) for digit in time if digit != ":"]))
        ava_digits.sort()

        # MM checking. Find the smallest greater number satisying the condition
        mm_val = int(mm)
        for i in range(len(ava_digits)):
            if ava_digits[i] >= 6:
                break
            for j in range(len(ava_digits)):

                if ava_digits[i] * 10 + ava_digits[j] > mm_val:
                    res_mm = ava_digits[i] * 10 + ava_digits[j]
                    if res_mm < 10:
                        res_mm = "0" + str(res_mm)
                    else:
                        res_mm = str(res_mm)
                    return ":".join([hh, res_mm])


        # if not: HH checkiung. Find the smallest greater number satisying the condition
        hh_val = int(hh)
        for i in range(len(ava_digits)):
            if ava_digits[i] >= 3:
                break

            for j in range(len(ava_digits)):
                if ava_digits[i] == 2 and ava_digits[j] > 4:
                    break
                if ava_digits[i] * 10 + ava_digits[j] > hh_val:
                    res_hh = ava_digits[i] * 10 + ava_digits[j]
                    if res_hh < 10:
                        res_hh = "0" + str(res_hh)
                    else:
                        res_hh = str(res_hh)
                    return ":".join([res_hh, str(ava_digits[0]) + str(ava_digits[0])])
        # if both not. Return the smallest
        return ":".join([str(ava_digits[0]) + str(ava_digits[0]), str(ava_digits[0]) + str(ava_digits[0])])

