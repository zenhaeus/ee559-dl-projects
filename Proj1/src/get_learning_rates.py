import json


params = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
params_dict = {"learning_rates": params}

with open("../output/learning_rates/AuxNet2/params_dict.json", "w") as f:
    json.dump(params_dict, f)
