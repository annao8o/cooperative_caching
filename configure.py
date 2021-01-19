## simulator param
region_num = 4
type_num = 4
contents_num = 50
request_rate = 1000
req_time = 24
time_period = 24
update_period = 24
cache_capacity = 45
zipf_param = 1.0
interval = 1
user_num = 100

user_density = [[100,100,200,600],[100,200,600,100],[200,600,100,100],[600,100,100,200]]   # region 1: 평균 0번 100명, 1번 200명, 2번 300명, 3번 400명이 존재



folder_path = './data/'
req_file = 'requests(test__).pickle'
pref_file = 'region(test__).pickle'
history_file = 'history(test__).pickle'
predicted_file = 'predict(test__).pickle'


