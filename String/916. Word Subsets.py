class Solution:
    def wordSubsets(self, A, B):
        """
        :type A: List[str]
        :type B: List[str]
        :rtype: List[str]
        """
        b_dicts = collections.defaultdict(int)
        for x in B:
            for key, count in collections.Counter(x).items():
                b_dicts[key] = max(b_dicts[key], count) 
            
        # for each word a in A, we need to check a_dict with b_dict
        res = []
        for x in A:
            dict_a = collections.Counter(x)
            if all(dict_a[i] >= b_dicts[i] for i in b_dicts):
                res.append(x)

        
        return res
            
            
            
            
            
        