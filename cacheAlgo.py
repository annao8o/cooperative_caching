import numpy as np
from operator import itemgetter

class CacheAlgo:
    def __init__(self, id, region):
        self.id = id
        self.region = region
        self.capacity = None
        self.cached_contents = list()
        self.p_dict = dict()
        self.rep_method = None
        self.is_cluster = None
        self.rep_cnt = 0
        self.LCU_num = 0
        self.pred = None

    def set_option(self, capacity, LCU_num, cluster_num, is_cluster=False, pred=False, rep=None):
        self.capacity = capacity
        self.LCU_num = LCU_num
        self.is_cluster = is_cluster
        self.rep_method = rep
        self.cached_contents = [[] for _ in range(cluster_num)]
        self.pred = pred

    def calc_capacity_per_cluster(self):
        density = self.region.user_density
        tmp = [(i, v) for i, v in enumerate(density)]
        max_lst = list()    #LCU개수만큼 가장 density가 높은 k 저장 
        max_value_lst = list()
        k_capacity = [0 for _ in range(len(density))]
            
        for i in range(self.LCU_num):
            max_idx = max(tmp, key=itemgetter(1))[0]
            max_value = density[max_idx]
            max_value_lst.append(max_value)
            max_lst.append((max_idx, max_value))
            tmp.remove((max_idx, max_value))
        sorted_max = sorted(max_lst, key=itemgetter(0))
        j = 0
        for c in range(len(self.cached_contents)):
            if j >= len(sorted_max):
                break
            else:
                if c == sorted_max[j][0]:
                    k_capacity[c] = round(self.capacity * sorted_max[j][1] / sum(max_value_lst))
                    j += 1
        return k_capacity
                    
    def check_capacity(self, k):
        is_full = False
        if self.is_cluster:
            k_capacity = self.calc_capacity_per_cluster()
            if len(self.cached_contents[k]) >= k_capacity[k]:
                is_full = True            
        else:
            if len(self.cached_contents[0]) >= self.capacity:
                is_full = True
                
        return is_full
                

    def have_content(self, content):
        self.p_dict[content] = self.p_dict.get(content, 0) + 1
        hit = False
        if content in [element for array in self.cached_contents for element in array]:
            hit = True
            
        return hit
        
    def place_content(self, k, data):    #k: user_type(cluster)
        if not self.is_cluster:
            k = 0

        if not self.check_capacity(k) :   #LCU is not full 
            self.cached_contents[k].append(content) #store
        else:   #LCU is full
            self.replace_content(k, content)    #replace content
                          

    def place_content2(self, global_popular):   #Globally popular data 저장
        data = global_popular
        self.cached_contents[0] = data[:self.capacity]
        return self.cached_contents

    def place_content3(self, regional_popular): #Regional popular data 저장
        data = regional_popular
        for k in range(len(self.cached_contents)):
            for content in data[k]:
                if self.check_capacity(k):
                    break
                else:
                    if content in [element for array in self.cached_contents for element in array]:
                        continue                
                    self.cached_contents[k].append(content)
        return self.cached_contents
        
    def place_content4(self, t, update_period):
        results = list()
        results = self.calc_p(t, update_period)
        print(results)
        self.cached_contents[0] = results[:self.capacity]
        return self.cached_contents
        

    def calc_p(self, t, update_period):
        multiple_lst = []
        idx = 0
        print("time", idx)
        for u_id in self.region.predicted_pref:
            while idx < t+update_period:
                tmp = list()
                for c in range(len(self.region.predicted_pref[u_id])):
                    # print(self.region.predicted_pref[u_id][c][idx])
                    tmp.append(np.array(self.region.predicted_pref[u_id][c][idx]))
                multiple_lst.append(tmp)
                idx += 1
            print(len(multiple_lst))
            print(len(multiple_lst[0]))
        result = [np.mean(k) for k in zip(*multiple_lst)]

        idx_p = list()
        for i, v in enumerate(result):
            idx_p.append((i, v))

        # sort
        idx_p.sort(key=lambda t:t[1], reverse=True)
        return [e[0] for e in idx_p]
    
        
    def replace_content(self, content):
        evict = None
        
        if self.rep_method == 'None':
            return
        
        else:
            if self.rep_method == 'FIFO':
                evict = self.cached_contents[k][0]
                self.cached_contents[k].remove(evict)
                self.cached_contents[k].append(content)
                
            elif self.rep_method == 'LFU':
                return
            
            elif self.rep_method == 'PREDICTION':
                elements = [element for array in self.cached_contents for element in array]
                for i, v in enumerate(self.region.priority_T):
                    if v in elements:
                        least = v
                        least_priority = i
                        break
                    
                content_priority = self.region.priority_T.index(content)

                if content_priority > least_priority:
                    evict = least
                    self.cached_contents[k].remove(evict)   #k와의 연관성이 없는데,,,?
                    self.cached_contents[k].append(content)                
            
            
            self.rep_cnt += 1
            print("{} | Evict: {} ---> New: {}".format(self.id, evict, content))




    
