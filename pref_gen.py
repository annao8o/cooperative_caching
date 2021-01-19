import numpy as np
import copy
from sklearn.cluster import KMeans
import random
from configure import *
from lstm import *
import pandas as pd
from pandas import DataFrame


def add_noise(values, noise_value):   # array에 noise를 더함
    if type(values) == list:
        values = np.array(values)

    values = values / values.sum()
    dev = np.random.random(values.shape)
    dev = dev - dev.sum() / len(dev)
    dev = dev * noise_value * 2  # dev 범위 = (-0.5, 0.5) -> (-noise_value, noise_value)

    result = values + dev  # add noise

    result[np.where(result < 0.0)] = 0.0
    result[np.where(result > 1.0)] = 1.0

    result = result / result.sum()  # normalize

    return result
    

def make_cluster(users, cluster_num):
        p_vectors = list()
        for user in users:
            p_vectors.append(user.pref_vec[0])                                                                 

        n_cluster = cluster_num

        kmeans = KMeans(n_clusters=n_cluster, init='random', algorithm='auto')
        kmeans.fit(p_vectors)
        cluster_centers = kmeans.cluster_centers_

        for i in range(len(users)):
            users[i].set_cluster(kmeans.labels_[i])

        
class PrefGenerator:
    def __init__(self, id):
        self.id = id
        self.users = list()
        self.type_lst = None  # store preference type
        self.num_type = None
        self.contents_num = None
        self.algo_lst = list()
        self.total_request = 0
        self.popularity = None  #regional popularity 확인
        self.user_density = None
        self.priority_T = None
        self.cdf = None     # region 내 users의 cdf 평균
        self.pdf = None
        self.predicted_pref = dict()
    
    def make_pref_type(self, num_type, contents_num, z_val):
        self.num_type = num_type
        self.contents_num = contents_num
        self.type_lst = np.zeros((num_type, contents_num), dtype=np.float_)
            
        temp = np.power(np.arange(1, contents_num + 1), -z_val)
        denominator = np.sum(temp)
        pdf = temp / denominator

        h_lst = [i for i in range(contents_num)]
        bound = contents_num // num_type
        for type_idx in range(num_type):
            unused_lst = [i for i in range(contents_num)]
            for i in range(bound):
                c_i = np.random.choice(h_lst)
                h_lst.remove(c_i)
                unused_lst.remove(c_i)
                self.type_lst[type_idx, c_i] = pdf[i]
            np.random.shuffle(unused_lst)
            for i in range(bound, contents_num):
                self.type_lst[type_idx, unused_lst[i-bound]] = pdf[i]
        return self.type_lst

    def set_pref_type(self, type_lst):
        self.type_lst = type_lst
        self.num_type = len(type_lst)
        self.contents_num = len(type_lst[0, :])
    
    def make_user_pref(self, type_weight=None, dev_val=0.001):  # type_weight: 타입과의 유사도? p = w1*t1 + w2*t2 + ...  , out_type: 0=cdf, 1=pdf
        if type_weight is None:
            type_weight = np.random.random((self.num_type, 1))

        type_weight = type_weight / type_weight.sum()   # normalization

        # result = self.type_lst * type_weight
        # result = result.sum(axis=0)    # pdf
        user_type = np.random.choice(len(type_weight), p=type_weight.flatten())

        result = add_noise(self.type_lst[user_type, :], dev_val)  # add noise for user
        
        return user_type, result, np.r_[0.0, np.cumsum(result)]  # return pdf, cdf

    def add_algo(self, algo):
        if type(algo).__name__ == 'CacheAlgo':
            self.algo_lst.append(algo)
            # print('Success to add algo')
        else:
            print('wrong algo class')
            
    def add_user(self, user):
        self.users.append(user)


    def prediction(self, data):
        for u in self.users:
            #dataframe으로 변환
            for t in range(len(data[u.id])):
                if t == 0:
                    df = pd.DataFrame([data[u.id][t][1]], columns = range(0, contents_num))
                else:
                    df.loc[t] = data[u.id][t][1]
            result = lstm_ex(df.values, contents_num, time_period)
            print(result)
            self.predicted_pref[u.id] = result
            print("prediction of region {}) user {} is done".format(self.id, u.id))
            
            
    
    def calc_p_k(self, users):
        tmp = [0 for _ in range(self.contents_num)]
        
        for u in users:
            for i in range(len(tmp)):
                tmp[i] += u.pref_vec[0][i]

        p_k = [element / len(users) for element in tmp]

        idx_p_tuple = list()
        tmp = list()
        for i, v in enumerate(p_k):
            idx_p_tuple.append((i, v))

        # sort
        idx_p_tuple.sort(key=lambda t: t[1], reverse=True)
        popularity_sorted = [e[0] for e in idx_p_tuple]
        return popularity_sorted


    def get_popular_contents(self, is_cluster):
        p_k = [[] for _ in range(len(self.type_lst))]
        for i in range(len(self.type_lst)):
            user_lst = list()
            for u in self.users:
                #if u.user_type == i:
                user_lst.append(u)
            p_k[0] = self.calc_p_k(user_lst)

        return p_k
        

    def request(self, content):
##        f = np.random.random()
##        content = np.searchsorted(self.cdf, f) - 1
##        self.popularity[content] += 1
        self.total_request += 1
        hit_lst = list()

        for algo in self.algo_lst:
            hit = algo.have_content(content)
            if hit:
##                print("{} >> hit!".format(algo.id))
                hit_lst.append(1)
            else:
##                print("{} >> no hit!".format(algo.id))
                hit_lst.append(0)
                algo.replace_content(content)

        return hit_lst
    
        
##        # print('requests content (id: {})'.format(content_id))
####        print('region ', self.id)
##        self.popularity[content] += 1
##        self.total_request += 1
##        hit_lst = list()
##
##        for algo in self.algo_lst:
##            hit = algo.have_content(content)
##            if hit:
####                print("{} >> hit!".format(algo.id))
##                hit_lst.append(1)
##            else:
####                print("{} >> no hit!".format(algo.id))
##                hit_lst.append(0)
##                algo.place_content(user_type, content)
##
##        return hit_lst
##    
class User:
    def __init__(self, id):
        self.id = id
        self.pref_vec = None    #(pdf, cdf)
        self.user_type = None
        # self.cluster = None
        self.region = None
        self.pref_data = list()  #preference 변화 record

    def set_char(self, region, user_type, pdf, cdf):
        self.region = region
        self.pref_vec = (pdf.ravel(), cdf.ravel())
        self.user_type = user_type
        self.pref_data.append(self.pref_vec[0])

    def request(self):
        f = np.random.random()
        content = np.searchsorted(self.pref_vec[1], f) - 1

        hit_lst = self.region.request(self.user_type, content)
        
        return content, hit_lst

##    def set_cluster(self, cluster_id):
##        self.cluster = cluster_id
    
class Zipf:
    def __init__(self):
        self.pdf = None
        self.cdf = None
        
##    def set_env(self, expn, num_contents):
##	temp = np.power(np.arange(1, num_contents+1), -expn)
##	zeta = np.r_[0.0, np.cumsum(temp)]
##	# zeta = np.r_[0.0, temp]
##	self.pdf = [x / zeta[-1] for x in temp]
##	self.cdf = [x / zeta[-1] for x in zeta]
        
    def set_env(self, users):
        pdf_lst = []
        cdf_lst = []
        for u in users:
            pdf_lst.append(np.array(u.pref_vec[0]))
            cdf_lst.append(np.array(u.pref_vec[1]))
        self.pdf = [np.mean(k) for k in zip(*pdf_lst)]
        self.cdf = [np.mean(k) for k in zip(*cdf_lst)]
##        print(self.pdf)
        return self.pdf, self.cdf

    def get_sample(self, size=None):
        if size is None:
            f = random.random()
        else:
            f = np.random.random(size)
        v = np.searchsorted(self.cdf, f)
        samples = [t-1 for t in v]
        return samples    
