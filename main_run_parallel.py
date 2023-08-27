#!/usr/bin/env python
# Created by "Thieu" at 17:49, 15/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
import concurrent.futures as parallel
from metacluster import get_dataset, MetaCluster


def execute_model(data_name):
    PATH_SAVE = "history_100"
    EPOCH = 100
    POP_SIZE = 50
    data = get_dataset(data_name)
    data.X, scaler = data.scale(data.X, method="MinMaxScaler")

    list_optimizer = ("SADE", "OriginalSSO", "OriginalAO", "ModifiedEO", "OriginalGSKA", "BaseSARO", "OriginalSOA", "AugmentedAEO", "OriginalRUN", "OriginalINFO")
    list_paras = [
        {"name": "SADE", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "SSO", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "AO", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "M-EO", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "GSKA", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "IQSA", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "SOA", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "AAEO", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "RUN", "epoch": EPOCH, "pop_size": POP_SIZE},
        {"name": "INFO", "epoch": EPOCH, "pop_size": POP_SIZE},
    ]
    list_obj = ["MSEI", "CHI", "RTS", "ARS"]
    list_metric = ["BHI", "DBI", "DI", "SI", "RSI", "FMS", "HS", "CS", "VMS", "KS"]

    model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=10)
    model.execute(data, cluster_finder="all_majority", list_metric=list_metric, save_path=PATH_SAVE, verbose=False)
    model.save_boxplots()
    model.save_convergences()


if __name__ == '__main__':
    time_start = time.perf_counter()
    list_datasets = ["aniso", "balance", "blobs", "circles", "smiley", "liver", "varied", "Glass", "Zoo", "Vowel"]
    N_CPUS_RUN = len(list_datasets)
    with parallel.ProcessPoolExecutor(N_CPUS_RUN) as executor:
        executor.map(execute_model, list_datasets)
    print(f"Experiment DONE: {time.perf_counter() - time_start} seconds")
