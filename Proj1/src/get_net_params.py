import json
from model import *


params = [16, 32, 64, 128, 256, 512]
params_dict = {}
count = 0
for i in params:
    for j in params:
        for k in params:
            for t in params:
                n_params = AuxNet1(i, j, k, t).count_params()
                if 65e+3 < n_params < 75e+3:
                    params_dict[count] = [i, j, k, t]
                    print(i, j, k, t, ":", n_params)
                    count += 1

with open("../output/net_params/AuxNet1/params_dict.json", "w") as f:
    json.dump(params_dict, f)
