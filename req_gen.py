import pickle
import numpy as np
import random
import os.path
from pref_gen import *
from configure import *

##def make_requests(region_lst, arrival_rate, interval, total_time, zipf_lst):
##    req_lst = list()
##    t = 0
##    while t < total_time:
##        req_num = np.random.poisson(arrival_rate, size = len(region_lst))
##        for r_idx in range(len(region_lst)):
##            cnt_dict = dict()
##            if req_num[r_idx] != 0:
##                n = req_num[r_idx]
##                samples = zipf_lst[r_idx].get_sample(size=n)
##                for sample in samples:
##                    cnt_dict[sample] = cnt_dict.get(sample, 0)+1
##                    req_lst.append((t, region_lst[r_idx].id, sample)) #(t, region_id, content)
##            a = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=True)
####            print(a)
##
####                for i in range(req_num[r_idx]):
####                    req_lst.append((t, region_lst[r_idx].id, zipf_lst[r_idx].get_sample())) #(t, region_id, content)
##        t += interval
##    req_lst.sort(key=lambda x: x[0])
##    return req_lst

def make_requests(region_lst, request_rate, zipf_lst):
    req_lst = [[] for _ in range(len(region_lst))]

    req_num = np.random.poisson(request_rate, size = len(region_lst))
    for r_idx in range(len(region_lst)):
        if req_num[r_idx] != 0:
            n = req_num[r_idx]
            samples = zipf_lst[r_idx].get_sample(size=n)
            req_lst[r_idx] = samples
    return req_lst

def save_requests(file, req_lst):
    with open(os.path.join(file), 'wb') as f:
        pickle.dump(req_lst, f)
    print("Success to save the requests")
    
def load_requests(file):
    with open(os.path.join(file), 'rb') as f:
        print("Success to load the requests")
        return pickle.load(f)

def make_preference(region_num, type_num, contents_num, zipf_param, user_density, user_num):
    region_lst = [PrefGenerator(i) for i in range(region_num)]
    base_type_lst = region_lst[0].make_pref_type(type_num, contents_num, zipf_param)   #make type
    region_lst[0].user_density = user_density[0]
    for i in range(1, region_num):
        region_lst[i].set_pref_type(base_type_lst)
        region_lst[i].user_density = user_density[i]

    pref_lst = list()
    for region in region_lst:
        d = dict()  #user preference history
        for i in range(user_num):
            user = User(i)
            record_history(region, i, d)
            # user_type, pdf, cdf = region.make_user_pref(type_weight=add_noise(region.user_density, 0.0001))
            # user.set_char(region, user_type, pdf, cdf)
            region.add_user(user)
        pref_lst.append(d)
    return region_lst, pref_lst

def record_history(region, user_id, d):
    for t in range(time_period):
        user_type, pdf, cdf = region.make_user_pref(type_weight=add_noise(region.user_density, 0.0001))
        if user_id in d.keys():
            d[user_id].append((user_type, pdf, cdf))
        else:
            d[user_id] = [(user_type, pdf, cdf)]
    
    
def save_preference(file, pref):
    with open(os.path.join(file), 'wb') as f:
        pickle.dump(pref, f)
    print("Success to save the preference")

def load_preference(file):
    with open(os.path.join(file), 'rb') as f:
        print("Success to load the preference")
        return pickle.load(f)

    
