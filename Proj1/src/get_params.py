import json
import model as m


def get_net_params(name="BasicNet1"):
    params = [16, 32, 64, 128, 256, 512]
    params_dict = {}
    count = 0
    for i in params:
        for j in params:
            for k in params:
                for t in params:
                    n_params = getattr(m, name)(i, j, k, t).count_params()
                    if 65e+3 < n_params < 75e+3:
                        params_dict[count] = [i, j, k, t]
                        print(i, j, k, t, ":", n_params)
                        count += 1

    with open("../output/net_params/{}/params_dict.json".format(name), "w") as f:
        json.dump(params_dict, f)

def get_learning_rates(name="AuxNet2"):
    params = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    params_dict = {"learning_rates": params}

    with open("../output/learning_rates/{}/params_dict.json".format(name), "w") as f:
        json.dump(params_dict, f)

def get_batch_sizes(name="AuxNet2"):
    params = [25, 50, 100, 200, 500, 1000]
    params_dict = {"batch_sizes": params}

    with open("../output/batch_sizes/{}/params_dict.json".format(name), "w") as f:
        json.dump(params_dict, f)