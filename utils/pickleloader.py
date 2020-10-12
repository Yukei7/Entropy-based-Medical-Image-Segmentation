#!/usr/bin/env python

import pickle


def load_pickle(path):
    with open(path, 'rb') as file:
        pkl = pickle.load(file)
    return pkl


def dump_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)
