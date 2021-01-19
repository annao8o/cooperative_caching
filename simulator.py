from req_gen import *
from cacheAlgo import *
from configure import *
from pref_gen import *
from req_gen import *
import os.path

def calc_p(users): 
    multiple_lst = []
    for u in users:
        multiple_lst.append(np.array(u.pref_vec[0]))
    result = [np.mean(k) for k in zip(*multiple_lst)]

    idx_p = list()
    for i, v in enumerate(result):
        idx_p.append((i, v))

    # sort
    idx_p.sort(key=lambda t:t[1], reverse=True)
    return [e[0] for e in idx_p]


def get_popular(r, contents_num):
    results = list()
    if isinstance(r, list): #global
        region_lst = r
        user_lst = list()
        for region in region_lst:
            for u in region.users:
                user_lst.append(u)
        results = calc_p(user_lst)
    else:
        region = r
        for i in range(4):
            user_lst = list()
            for user in region.users:
                if user.user_type == i:
                    user_lst.append(user)
            results.append(calc_p(user_lst))
        
    return results

def simulation(region_lst, pref_lst, type_num, contents_num, request_rate, time_period):
    hit_result = [[] for _ in range(len(region_lst))]
    all_req = dict()
    every_time_req = dict()
    zipf_lst = [Zipf() for _ in range(region_num)]
    
    for r in range(len(region_lst)):
        hit_result[r] = [0 for _ in range(len(region_lst[0].algo_lst))]
                          
##    global_popular = get_popular(region_lst, contents_num)    
##    for region in region_lst:
##        regional1 = calc_p(region.users)
##        regional2 = get_popular(region, contents_num)
##
##        for algo in region.algo_lst:
##            if algo.id == 'Global':
####                print('Global', algo.place_content2(global_popular), '\n')
##                algo.place_content2(global_popular)
##            else:
##                if algo.is_cluster:
####                    print(region.id, algo.id, algo.place_content3(regional2), '\n')
##                    algo.place_content3(regional2)
##                else:
####                    print(region.id, algo.id, algo.place_content2(regional1), '\n')
##                    algo.place_content2(regional1)  
    t = 0
    req_idx = 0       
    
    while t < time_period:
        for r in range(len(pref_lst)):  #user setting
            for user in region_lst[r].users:
                # print(pref_lst[r][user.id][t])
                user_type, pdf, cdf = pref_lst[r][user.id][t]
                user.set_char(region_lst[r], user_type, pdf, cdf)
            region_lst[r].pdf, region_lst[r].cdf = zipf_lst[r].set_env(region_lst[r].users)
        ##        print("------------------\ntime {}".format(t))

        if t % update_period == 0:
            print("updated...")
            global_popular = get_popular(region_lst, contents_num)    
            for region in region_lst:
                regional = calc_p(region.users)
                for algo in region.algo_lst:
                    if algo.id == 'Global':
                      print('Global', algo.place_content2(global_popular), '\n')
##                      algo.place_content2(global_popular)
                    else:
                        if not algo.pred:
                            print('Regional/non_pred', algo.place_content2(regional), '\n')
##                            algo.place_content2(regional)
                        else:
                            print('Regional/pred', algo.place_content4(t, update_period), '\n')
##                            algo.place_content4(t, update_period)
                       
        
        # Requests
##        for z_idx in range(len(zipf_lst)):
##            region_lst[z_idx].pdf, region_lst[z_idx].cdf = zipf_lst[z_idx].set_env(region_lst[z_idx].users)
        
        req_lst = make_requests(region_lst, request_rate, zipf_lst)

        for r_idx in range(len(req_lst)):
            for content in req_lst[r_idx]:
                hit_lst = region_lst[r_idx].request(content)
                for i in range(len(hit_lst)):
                    hit_result[r_idx][i] += hit_lst[i]
                    
##        while req_idx <= len(req_lst):  
##            if req_lst[req_idx][0] == t:
##                region = region_lst[req_lst[req_idx][1]]
##                content = req_lst[req_idx][2]
##                hit_lst = region.request(content)
####                print("Request content{} in region{}\n".format(content, region.id))
##                for i in range(len(hit_lst)):
##                    hit_result[region.id][i] += hit_lst[i]
##                req_idx += 1
##            else:
##                break

        t += 1
    total_request = [region.total_request for region in region_lst]
    result = {'total_request': total_request, 'hit_count': hit_result, 'replacement_count': [[algo.rep_cnt for algo in region.algo_lst] for region in region_lst],
              'hit_ratio': [e/total_request[i] for i in range(len(hit_result)) for e in hit_result[i]] }
    print(result)

if __name__ == "__main__":
    
    ## Generate or Load users for all regions
    if os.path.isfile(folder_path+pref_file):
        print("Load the region...")
        region_lst = load_preference(folder_path+pref_file)
        pref_lst = load_preference(folder_path+history_file)
        
    else:
        print("Generate the region...")
        region_lst, pref_lst = make_preference(region_num, type_num, contents_num, zipf_param, user_density, user_num)
        for region in region_lst:
            region.prediction(pref_lst[region.id])
        save_preference(folder_path+pref_file, region_lst)
        save_preference(folder_path + history_file, pref_lst)
    
    
    '''
    ## Generate or Load requests
    if os.path.isfile(folder_path+req_file):
        print("Load the requests...")
        req_lst = load_requests(folder_path+req_file)
    else:
        ## Generate and set zipf for each region
        zipf_lst = [Zipf() for _ in range(region_num)]
        for z_idx in range(len(zipf_lst)):
            region_lst[z_idx].pdf, region_lst[z_idx].cdf = zipf_lst[z_idx].set_env(region_lst[z_idx].users)
            
        print("Generate the requests...")
        req_lst = make_requests(region_lst, request_rate, interval, req_time, zipf_lst)
        save_requests(folder_path+req_file, req_lst)
    '''
    for region in region_lst:
        print(len(region.predicted_pref[1]))
        # prediction
        # region.prediction(pref_lst[region.id])
        # print('Region{}\n[PDF]\n{}\n'.format(region.id, region.pdf))

        algo1 = CacheAlgo('algo1', region)
        algo1.set_option(cache_capacity, 1, type_num, is_cluster = False, pred = True, rep='None')

        algo2 = CacheAlgo('algo2', region)
        algo2.set_option(cache_capacity, 1, type_num, is_cluster = False, pred = False, rep='None')
        
        algo3 = CacheAlgo('Global', region)
        algo3.set_option(cache_capacity, 1, type_num, is_cluster = False, pred = False, rep = 'None')
        
        region.add_algo(algo1)
        region.add_algo(algo2)
        region.add_algo(algo3)

    # print(pref_lst)
    simulation(region_lst, pref_lst, type_num, contents_num, request_rate, time_period)
