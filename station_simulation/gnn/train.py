import sys

sys.path.append("../../")

import numpy as np
import simulator
import config
import copy
import argparse
import joblib
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm

from Infections.Agent import Symptoms, Category
from pprint import pprint
import os.path
from os import path
from ML.mlutils import progressbar, get_loss_function


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# -- weighting
CLASS_WEIGHTS = {0: 1.0, 1: 500.0}  # misclassifying 1 is worse
LINEAR_DISCOUNT_FACTOR = 500.0
# -- dataset
SIMULATED_TIME_DAYS = 100
TRAINING_TIME_DAYS = 40
# --
BATCH_SIZE = 10
DATASET_SIZE_IN_BATCHES = 10
EPOCHS_BEFORE_REGENERATE = 10
# -- how much training
REPETIONS = 1
EPOCHS = 10**6  # do not stop
STEPS_PER_EPOCH = DATASET_SIZE_IN_BATCHES  # ok?

# VALIDATION_STEPS = 2


import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm


# ===================================

config.T = TRAINING_TIME_DAYS
NETWORK_FILE = f"my_neural_network-{config.T}-{config.N}-new.h5"
AUTOMATIC_RESUME = True

# EXPEXTED_TEST_PER_DAY = 3
# EXPEXTED_QUARANTINES_PER_DAY = 1.5
# QUARANTINE_DAYS = 7

# testing with small N
EXPEXTED_TEST_PER_DAY = config.N / 5
EXPEXTED_QUARANTINES_PER_DAY = config.N / 10
QUARANTINE_DAYS = 7


class DataGenerator(Sequence):
    def __init__(self, batch_size, dataset_size, epochs_before_regenerate, usetest):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.epochs_before_regenerate = epochs_before_regenerate
        self.simulation_generator = SimulationGenerator(
            traces_in_dataset=batch_size * dataset_size, usetest=usetest
        )
        self.current_epoch = 0
        # for shuffling and setpping
        self.indices = np.arange(batch_size * dataset_size)
        # our dataset
        self.__generate_dataset()

    def __len__(self):
        return self.dataset_size

    def __generate_dataset(self):
        print("Generate new dataset")
        # get traces for dataset
        self.x, self.y, self.weights = self.simulation_generator.generate_traces()
        # reshape them
        traces_nr = self.batch_size * self.dataset_size
        self.x = np.array(self.x).reshape(traces_nr, config.T, OBS_LENGTH)
        self.y = np.array(self.y).reshape(traces_nr, config.T, 1)
        self.weights = np.array(self.weights).reshape(traces_nr, config.T, 1)
        # print("exit")

    def __getitem__(self, idx):
        # print(f'get {idx}')
        inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        batch_weights = self.weights[inds]
        return batch_x, batch_y, batch_weights

    def on_epoch_end(self):
        # print('epoch end')
        self.current_epoch += 1
        if self.current_epoch >= self.epochs_before_regenerate:
            # generate new dataset
            self.__generate_dataset()
            self.current_epoch = 0
        else:
            # just reshuffle
            print("Shuffling dataset")
            np.random.shuffle(self.indices)
            pass


def plot_run(
    model,
    usetest,
    nogui=False,
    filename="learn_simple-out.png",
    T=SIMULATED_TIME_DAYS,
    initial_infections={"T_range": (0, 10), "N_range": (0, 2)},
):
    print("Plotting run...")

    config.T = T

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
    ) = get_run(usetest=usetest, initial_infections=initial_infections)
    # prepare info for plotting
    info = (
        np.array(symptom_none) * 0.2
        + np.array(symptom_mild) * 0.5
        + np.array(symptom_severe) * 0.8
    ) * np.array(spreading)
    # prediction for all agents
    all_pred = None
    for i in progressbar(range(config.N), "Predicting agents: "):
        # bring agent i into position 0
        swap_agents(
            i,
            0,
            contactgraphs,
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
            test_positive,
            test_negative,
        )
        # build input
        x = np.concatenate(
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
                contactgraphs2matrix(contactgraphs),
            ),
            axis=1,
        )
        T_data = [x]
        # DATA_SIZE is 1 since has only 1 run to predict
        T_data = np.array(T_data).reshape(1, config.T, OBS_LENGTH)
        # predict agent
        pred = predict_agent0(
            model=model, input_vector=T_data, window_size=TRAINING_TIME_DAYS
        )
        # add to all predictions
        if all_pred is None:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=1)
        # swap back! Otherwise this will mess with the vectors that are diplayed next!
        swap_agents(
            i,
            0,
            contactgraphs,
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
            test_positive,
            test_negative,
        )

    # print(info)
    plt.figure(figsize=(9, 9))

    # spreading
    plt.subplot(1, 2, 1)
    plt.imshow(info, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    # tested
    my_reds = copy.copy(cm.get_cmap("Reds"))
    my_reds.set_under("k", alpha=0)
    plt.imshow(
        0.5 * np.array(test_positive),
        cmap=my_reds,
        vmin=0.1,
        vmax=1,
        interpolation="none",
    )
    my_greens = copy.copy(cm.get_cmap("Greens"))
    my_greens.set_under("k", alpha=0)
    plt.imshow(
        0.5 * np.array(test_negative),
        cmap=my_greens,
        vmin=0.1,
        vmax=1,
        interpolation="none",
    )

    # predicted
    plt.subplot(1, 2, 2)
    plt.imshow(1 - all_pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    plt.tight_layout()
    print("[done plotting]")
    if not nogui:
        plt.show(block=True)
    else:
        plt.savefig(filename, dpi=300)


# =========================================


# main
if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser(description="Learn infection models.")
    parser.add_argument("--nogui", action="store_true", help="disable the gui")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start training from scratch and reset the NN file",
    )
    parser.add_argument(
        "--onlyplot", action="store_true", help="only plot, no training"
    )
    args = parser.parse_args()

    if args.fresh:
        print("Warning: will overwrite NN file if it exists.")
        AUTOMATIC_RESUME = False

    # turn off standard output
    print("Learning started...")
    config.plotting = False
    config.output_singleruns = False
    config.output_summary = False
    config.T = TRAINING_TIME_DAYS  # Important: train on shorter timeframe
    # choose test
    usetest = mytests[1]
    # number of runs to get
    N_runs = 1
    model = resume_NN(NETWORK_FILE, AUTOMATIC_RESUME)

    if not args.onlyplot:
        # train
        try:
            for i in range(REPETIONS):
                # train
                print("Fitting...")
                mydatagenerator = DataGenerator(
                    batch_size=BATCH_SIZE,
                    dataset_size=DATASET_SIZE_IN_BATCHES,
                    epochs_before_regenerate=EPOCHS_BEFORE_REGENERATE,
                    usetest=usetest,
                )
                history = model.fit(
                    mydatagenerator,
                    # use_multiprocessing=True, workers=2,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    # validation_data=data_generator(),
                    # validation_steps=VALIDATION_STEPS,
                    callbacks=[
                        ModelCheckpoint(
                            filepath=NETWORK_FILE,
                            save_weights_only=True,
                            monitor="loss",
                            mode="min",
                            save_best_only=False,
                        )
                    ],
                )
                print("[done fitting]")
                # save
                # print(f'Save after {i+1} repetitions...')
                # model.save(NETWORK_FILE)
                # print('[done saving]')
                # plot
                # print(history.history)
                plt.figure(figsize=(9, 9))
                labels = [
                    "loss",
                ]
                for lab in labels:
                    plt.plot(history.history[lab], label=f"{lab} model")
                plt.yscale("log")
                plt.legend()
                if not args.nogui:
                    # plt.show(block=True)
                    pass
                else:
                    plt.savefig(f"hist-{config.T}-{config.N}.png", dpi=300)

        except KeyboardInterrupt as e:
            print(f"Stopped training by user.")

    # plot
    try:
        plot_run(model, usetest, nogui=args.nogui)
    except KeyboardInterrupt as e:
        print(f"Stopped plotting by user.")
