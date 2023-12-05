import numpy as np
import simulator
import config as config
import joblib
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm

from Infections.Agent import Symptoms, Category
import os.path
from os import path
from ML.mlutils import progressbar


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import random


# ===================================

# -- multiprocessing
JOBNUMBER = -1  # -1 means all CPUs
MULTIPROCESSING_DATAGEN = True

# -- dataset
SIMULATED_TIME_DAYS = 100
TRAINING_TIME_DAYS = 40

config.T = TRAINING_TIME_DAYS

# EXPEXTED_TEST_PER_DAY = 3
# EXPEXTED_QUARANTINES_PER_DAY = 1.5
# QUARANTINE_DAYS = 7

# testing with small N
EXPEXTED_TEST_PER_DAY = config.N / 5
EXPEXTED_QUARANTINES_PER_DAY = config.N / 10
QUARANTINE_DAYS = 7


# tests:
mytests = {}
mytests[0] = {"type": "TestNull", "parameters": {}}
mytests[1] = {
    "type": "TestRandom",
    "parameters": {
        "prob_test": EXPEXTED_TEST_PER_DAY / config.N,
        "prob_quarantine": EXPEXTED_QUARANTINES_PER_DAY / config.N,
        "quarantine_days": QUARANTINE_DAYS,
    },
}

# observation: 12 fields + contact graph
OBS_LENGTH = (12 * config.N) + (config.N * config.N)


def boolvec2str(vec):
    return "".join(list(map(lambda b: "1" if b else "0", vec)))


def get_properties(state_vec, infex):
    #  12 observables + observable contact graph
    symptom_none = []
    symptom_mild = []
    symptom_severe = []
    symptom_dead = []
    category_MED = []
    category_NUR = []
    category_ADM = []
    category_PAT = []
    working = []
    quarantined = []
    test_positive = []
    test_negative = []
    contactgraphs = []
    # non observable
    infected = []
    spreading = []
    # convert
    current_T = len(state_vec)
    current_N = len(state_vec[0])
    for t in range(current_T):
        state_dict = state_vec[t]
        vec = [state_dict[agent_id] for agent_id in range(current_N)]

        # symptom classes (1-hot)
        symptom_none += [list(map(lambda x: x["symptoms"] is Symptoms.NONE, vec))]
        symptom_mild += [list(map(lambda x: x["symptoms"] is Symptoms.MILD, vec))]
        symptom_severe += [list(map(lambda x: x["symptoms"] is Symptoms.SEVERE, vec))]
        symptom_dead += [list(map(lambda x: x["symptoms"] is Symptoms.DEAD, vec))]

        # category classes (1-hot)
        category_MED += [list(map(lambda x: x["symptoms"] is Category.DOCTOR, vec))]
        category_NUR += [list(map(lambda x: x["symptoms"] is Category.NURSE, vec))]
        category_ADM += [list(map(lambda x: x["symptoms"] is Category.ADMIN, vec))]
        category_PAT += [list(map(lambda x: x["symptoms"] is Category.PATIENT, vec))]

        # desease progress properties
        infected += [list(map(lambda x: x["infected"], vec))]
        spreading += [list(map(lambda x: x["spreading"], vec))]
        working += [list(map(lambda x: x["working"], vec))]
        quarantined += [list(map(lambda x: x["quarantined"], vec))]

        # test results
        test_positive += [list(map(lambda x: x["testresult"] == True, vec))]
        test_negative += [list(map(lambda x: x["testresult"] == False, vec))]

        # get contact graphs
        contactgraphs += [infex.get_contactgraph_attime_withduration(t=t)]

    return (
        symptom_none,
        symptom_mild,
        symptom_severe,
        symptom_dead,
        category_MED,
        category_NUR,
        category_ADM,
        category_PAT,
        infected,
        spreading,
        working,
        quarantined,
        contactgraphs,
        test_positive,
        test_negative,
    )


def get_run(usetest, initial_infections={"T_range": (0, 10), "N_range": (0, 2)}):
    """
    run a simulation with the specified test.

    Arguments:
    Test as 'usetest'

    Returns:
    the run
    """
    results, infex, test, state_vec = simulator.run_sim(
        usetest, return_state_vec=True, initial_infections=initial_infections
    )
    #  12 observables
    symptom_none = []
    symptom_mild = []
    symptom_severe = []
    symptom_dead = []
    category_MED = []
    category_NUR = []
    category_ADM = []
    category_PAT = []
    working = []
    quarantined = []
    test_positive = []
    test_negative = []
    # contact graphs
    contactgraphs = []
    # non observable
    infected = []
    spreading = []
    # get properties
    (
        symptom_none,
        symptom_mild,
        symptom_severe,
        symptom_dead,
        category_MED,
        category_NUR,
        category_ADM,
        category_PAT,
        infected,
        spreading,
        working,
        quarantined,
        contactgraphs,
        test_positive,
        test_negative,
    ) = get_properties(state_vec, infex)
    return (
        symptom_none,
        symptom_mild,
        symptom_severe,
        symptom_dead,
        category_MED,
        category_NUR,
        category_ADM,
        category_PAT,
        infected,
        spreading,
        working,
        quarantined,
        contactgraphs,
        test_positive,
        test_negative,
    )


def contactgraphs2matrix(contactgraphs):
    """
    Input:
    contact graph

    Returns:
    matrix : (T, config.N, config.N) -> weight of contact
    """
    # init
    current_T = len(contactgraphs)
    current_N = len(contactgraphs[0].keys())
    matrix = np.zeros((current_T, current_N, current_N))
    # fill
    for t in range(current_T):
        for i in range(current_N):
            for j in contactgraphs[t][i].keys():
                matrix[t][i][j] = contactgraphs[t][i][j]
    return matrix


class SimulationGenerator(object):
    def __init__(self, traces_in_dataset, usetest):
        self.traces_in_dataset = traces_in_dataset
        self.usetest = usetest

    def generate_data(self):
        # switch to a larger time that is to be simulated
        config.T = SIMULATED_TIME_DAYS
        # simulate
        (
            symptom_none,
            symptom_mild,
            symptom_severe,
            symptom_dead,
            category_MED,
            category_NUR,
            category_ADM,
            category_PAT,
            infected,
            spreading,
            working,
            quarantined,
            contactgraphs,
            test_positive,
            test_negative,
        ) = get_run(usetest=self.usetest)
        # switch back to training time days
        config.T = TRAINING_TIME_DAYS
        # cut out TRAINING_TIME_DAYS from the SIMULATED_TIME_DAYS long run
        start_time = random.randrange(SIMULATED_TIME_DAYS - TRAINING_TIME_DAYS)
        end_time = start_time + TRAINING_TIME_DAYS
        # cutting
        symptom_none = symptom_none[start_time:end_time]
        symptom_mild = symptom_mild[start_time:end_time]
        symptom_severe = symptom_severe[start_time:end_time]
        symptom_dead = symptom_dead[start_time:end_time]
        category_MED = category_MED[start_time:end_time]
        category_NUR = category_NUR[start_time:end_time]
        category_ADM = category_ADM[start_time:end_time]
        category_PAT = category_PAT[start_time:end_time]
        infected = infected[start_time:end_time]
        spreading = spreading[start_time:end_time]
        working = working[start_time:end_time]
        quarantined = quarantined[start_time:end_time]
        contactgraphs = contactgraphs[start_time:end_time]
        test_positive = test_positive[start_time:end_time]
        test_negative = test_negative[start_time:end_time]
        # symptom_none : [0 .. config.T - 1] [0 .. config.N - 1] -> {0,1}
        #   etc.
        # labels: (T, #nodes, #features)

        node_labels = np.stack(
            (
                symptom_none,
                symptom_mild,
                symptom_severe,
                symptom_dead,
                category_MED,
                category_NUR,
                category_ADM,
                category_PAT,
                working,
                quarantined,
                test_positive,
                test_negative,
            ),
            axis=-1,
        )
        # graph: (T, #nodes, #nodes)
        graph = contactgraphs2matrix(contactgraphs)
        spreading = np.array(spreading)
        #
        # speading : [0 .. config.T - 1] [0 .. config.N - 1] -> {0,1}
        #   if agent 0 is spreading at time t
        return graph, node_labels, spreading

    def generate_traces(self):
        graphs = []
        node_labels = []
        spreading = []

        if MULTIPROCESSING_DATAGEN:
            # results = Parallel(n_jobs=JOBNUMBER)( delayed(self.generate_data)() for i in range(self.traces_in_dataset) )
            with tqdm_joblib(
                tqdm(
                    desc=f"Dataset with {joblib.cpu_count()} CPUs: ",
                    total=self.traces_in_dataset,
                )
            ) as progress_bar:
                results = Parallel(n_jobs=JOBNUMBER)(
                    delayed(self.generate_data)() for i in range(self.traces_in_dataset)
                )

            for i in range(self.traces_in_dataset):
                graphs.append(results[i][0])
                node_labels.append(results[i][1])
                spreading.append(results[i][2])
        else:
            for _ in progressbar(range(self.traces_in_dataset), "Dataset: "):
                graph, node_l, spreading = self.generate_data()
                graphs.append(graph)
                node_labels.append(node_l)
                spreading.append(spreading)
        # just convert to np-array
        graphs = np.array(graphs)
        node_labels = np.array(node_labels)
        spreading = np.array(spreading)
        return graphs, node_labels, spreading


if __name__ == "__main__":
    data_generator = SimulationGenerator(traces_in_dataset=100, usetest=mytests[1])
    graphs, node_labels, spreading = data_generator.generate_traces()
