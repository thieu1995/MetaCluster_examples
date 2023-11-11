

import pandas as pd
import time

EPOCH = 50
POP_SIZE = 20
N_TRIALS = 3

OPTIMIZERS = ["SSA", "GWO", "MVO", "PSO"]
DATA_LISTS = ["iris", "aggregation", "banknote", "balance", "ecoli", "smiley"]

META_OPTIMIZERS = ["OriginalSSA", "OriginalGWO", "OriginalMVO", "OriginalPSO"]
META_DATA_LISTS = ["Iris", "aggregation", "banknote", "balance", "ecoli", "smiley"]


def evo():
    from EvoCluster import EvoCluster

    LIST_TIMER = []
    for opt in OPTIMIZERS:
        t1_start = time.perf_counter()
        optimizer = [opt]
        objective_func = ["SSE"]
        dataset_list = DATA_LISTS #Select data sets from the list of available ones
        num_of_runs = N_TRIALS #Select number of repetitions for each experiment.
        params = {'PopulationSize': POP_SIZE, 'Iterations': EPOCH} #Select general parameters for all optimizers (population size, number of iterations)
        export_flags = {'Export_avg': True, 'Export_details': True, 'Export_details_labels': True,
                        'Export_convergence': True, 'Export_boxplot': True} #Choose your preferemces of exporting files

        ec = EvoCluster(optimizer,objective_func, dataset_list, num_of_runs, params, export_flags,
                        auto_cluster=True, n_clusters='supervised', labels_exist=True,metric='euclidean')
        ec.run()
        t1_final = time.perf_counter() - t1_start
        LIST_TIMER.append([opt, t1_final])

    df = pd.DataFrame(LIST_TIMER, columns=["Optimizer", "Run_Time"])
    df.to_csv("evocluster_1algorithm_Mdataset.csv")


def meta():
    from metacluster import get_dataset, MetaCluster

    LIST_TIMER = []
    for idx_opt, opt_class in enumerate(META_OPTIMIZERS):

        t2_start = time.perf_counter()
        for data_name in META_DATA_LISTS:
            data = get_dataset(data_name)
            data.X, scaler = data.scale(data.X, method="MinMaxScaler")
            list_optimizer = (opt_class, )
            list_paras = ({"name": OPTIMIZERS[idx_opt], "epoch": EPOCH, "pop_size": POP_SIZE}, )
            list_obj = ["SSEI",]
            list_metric = ["MIS", "NMIS", "RaS", "ARS", "HS", "CS", "VMS", "PrS", "ReS", "FmS"]

            model = MetaCluster(list_optimizer=list_optimizer, list_paras=list_paras, list_obj=list_obj, n_trials=N_TRIALS)
            model.execute(data, cluster_finder="all_majority", list_metric=list_metric, save_path="history/1aMd", verbose=False)
            model.save_boxplots()
            model.save_convergences()

        t2_final = time.perf_counter() - t2_start
        LIST_TIMER.append([OPTIMIZERS[idx_opt], t2_final])

        df = pd.DataFrame(LIST_TIMER, columns=["Optimizer", "Run_Time"])
        df.to_csv("metacluster_1algorithm_Mdataset.csv")

# evo()
meta()
