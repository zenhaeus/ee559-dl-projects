import json


params = [25, 50, 100, 200, 500, 1000]
params_dict = {"batch_sizes": params}

with open("../output/batch_sizes/AuxNet2/params_dict.json", "w") as f:
    json.dump(params_dict, f)