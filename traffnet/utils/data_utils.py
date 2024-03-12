import math
import os
import pathlib

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


def collect_fn_preSelPath(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.vstack(labels)


def traffNet_collect_fn(samples):
    graphSeqTimeWindows, labels = map(list, zip(*samples))
    graphSeq = []
    for graph in graphSeqTimeWindows:
        graphSeq.append(graph)
    batched_graph = dgl.batch(graphSeq)
    labels_array = np.array(labels)
    labels_tensor = torch.tensor(labels_array)
    return batched_graph, labels_tensor


def getDataGenerate(t, data_path):
    graph = dgl.load_graphs(f'{data_path}/htg_with_feat/Hetero_{t}.bin', [0])[0][0]
    label = graph.nodes['path'].data['pathNum']
    return (graph, label)


def getDataGenerateOneStep(idx, labels, horizon, data_path):
    label = labels[idx:idx + horizon].squeeze(axis=1)
    batchGraph = dgl.load_graphs(f'{data_path}/batch_htg_graph/batchHtg{idx}.bin', [0])[0][0]
    return (batchGraph, label)


def getSeqMaxSize(file_path):
    with open(f'{file_path}/path_dict.txt', 'r') as f:
        pathNodeDict = eval(f.readlines()[0])

    allPath = list(pathNodeDict.keys())
    seqMaxSize = max([len(path) for path in allPath])

    return seqMaxSize


class PsDataset(Dataset):
    def __init__(self,
                 datasetType,
                 timestamps,
                 window_width, out_len,
                 start_idx, data_path):
        self.data = []
        if datasetType == 'Train' or datasetType == 'Val':
            for t in tqdm(range(start_idx, start_idx + timestamps), desc=f'Process {datasetType} data'):
                self.data.append(getDataGenerate(t, data_path))
        elif datasetType == 'Test':
            for t in tqdm(range(start_idx, start_idx + timestamps - window_width - out_len),
                          desc=f'Process {datasetType} data'):
                self.data.append(getDataGenerate(t, data_path))
        else:
            raise ValueError("Invalid datasetType: {}".format(datasetType))

        self.len = len(self.data)

    def __getitem__(self, index):
        return (self.data[index][0], self.data[index][1])

    def __len__(self):
        return self.len


class FpDataset(Dataset):
    def __init__(self, datasetType,
                 timestamps,
                 window_width,
                 out_len,
                 start_idx, data_path):
        self.labels = np.load(f'{data_path}/label.npz')['arr_0']
        self.data = []
        if datasetType == 'Train' or datasetType == 'Val':
            for t in tqdm(range(start_idx, start_idx + timestamps), desc=f'Process {datasetType} data'):
                self.data.append(getDataGenerateOneStep(idx=t,
                                                        labels=self.labels,
                                                        horizon=out_len,
                                                        data_path=data_path))

        elif datasetType == 'Test':
            for t in tqdm(range(start_idx, start_idx + timestamps - window_width - out_len),
                          desc=f'Process {datasetType} data'):
                self.data.append(getDataGenerateOneStep(idx=t,
                                                        labels=self.labels,
                                                        horizon=out_len, data_path=data_path))
        else:
            raise ValueError("Invalid datasetType: {}".format(datasetType))

    def __getitem__(self, index):
        return (self.data[index][0], self.data[index][1])

    def __len__(self):
        return len(self.data)


def process_ps_dataset(logger, config):
    logger.info("Data start load")
    data_path = config["dataset"]["data_path"]
    train_split = config["dataset"]["train_split"]
    test_split = config["dataset"]["test_split"]
    val_split = config["dataset"]["val_split"]
    data_len = config["dataset"]["data_len"]
    timestampsTrain = int(data_len * train_split)
    timestampsVal = int(data_len * val_split)
    timestampsTest = int(data_len * test_split)
    train_start_idx = 0
    val_start_idx = train_start_idx + timestampsTrain
    test_start_idx = val_start_idx + timestampsVal
    train_dataset = PsDataset(datasetType='Train',
                              timestamps=timestampsTrain,
                              window_width=config["dataset"]["window_width"],
                              out_len=config["dataset"]["predict_len"],
                              start_idx=train_start_idx,
                              data_path=data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["train"]["path_select"]["batch_size"],
                                  shuffle=True,
                                  collate_fn=collect_fn_preSelPath)
    val_dataset = PsDataset(datasetType='Val',
                            timestamps=timestampsVal,
                            window_width=config["dataset"]["window_width"],
                            out_len=config["dataset"]["predict_len"],
                            start_idx=val_start_idx,
                            data_path=data_path)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["train"]["path_select"]["batch_size"],
                                shuffle=False,
                                collate_fn=collect_fn_preSelPath)
    test_dataset = PsDataset(datasetType='Test',
                             timestamps=timestampsTest,
                             window_width=config["dataset"]["window_width"],
                             out_len=config["dataset"]["predict_len"],
                             start_idx=test_start_idx,
                             data_path=data_path)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config["train"]["path_select"]["batch_size"],
                                 shuffle=False,
                                 collate_fn=collect_fn_preSelPath)
    actual_train_batch_size = train_dataloader.batch_size
    total_train_batches = math.ceil(len(train_dataset) / actual_train_batch_size)
    actual_val_batch_size = val_dataloader.batch_size
    total_val_batches = math.ceil(len(val_dataset) / actual_val_batch_size)
    actual_test_batch_size = test_dataloader.batch_size
    total_test_batches = math.ceil(len(test_dataset) / actual_test_batch_size)

    headers = ["", "batch_size", "batch_num"]
    data = [
        ["Train", actual_train_batch_size, total_train_batches],
        ["Val", actual_val_batch_size, total_val_batches],
        ["Test", actual_test_batch_size, total_test_batches]
    ]

    column_widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]

    table_width = sum(column_widths) + len(headers) * 3
    logger.info("-" * table_width)
    logger.info("|".join(headers[i].center(column_widths[i] + 2) for i in range(len(headers))))
    logger.info("-" * table_width)
    for row in data:
        logger.info("|".join(str(item).center(column_widths[i] + 2) for i, item in enumerate(row)))
    logger.info("-" * table_width)
    logger.info("Load data success!")
    return train_dataloader, val_dataloader, test_dataloader


def process_fp_dataset(logger, config):
    logger.info("Data start load")
    data_path = config["dataset"]["data_path"]
    train_split = config["dataset"]["train_split"]
    test_split = config["dataset"]["test_split"]
    val_split = config["dataset"]["val_split"]
    data_len = config["dataset"]["data_len"]
    timestampsTrain = int(data_len * train_split)
    timestampsVal = int(data_len * val_split)
    timestampsTest = int(data_len * test_split)
    train_start_idx = 0
    val_start_idx = train_start_idx + timestampsTrain
    test_start_idx = val_start_idx + timestampsVal
    train_dataset = FpDataset(datasetType='Train',
                              timestamps=timestampsTrain,
                              window_width=config["dataset"]["window_width"],
                              out_len=config["dataset"]["predict_len"],
                              start_idx=train_start_idx,
                              data_path=data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config["train"]["path_select"]["batch_size"],
                                  shuffle=True,
                                  collate_fn=traffNet_collect_fn)
    val_dataset = FpDataset(datasetType='Val',
                            timestamps=timestampsVal,
                            window_width=config["dataset"]["window_width"],
                            out_len=config["dataset"]["predict_len"],
                            start_idx=val_start_idx,
                            data_path=data_path)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config["train"]["path_select"]["batch_size"],
                                shuffle=False,
                                collate_fn=traffNet_collect_fn)
    test_dataset = FpDataset(datasetType='Test',
                             timestamps=timestampsTest,
                             window_width=config["dataset"]["window_width"],
                             out_len=config["dataset"]["predict_len"],
                             start_idx=test_start_idx,
                             data_path=data_path)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=config["train"]["path_select"]["batch_size"],
                                 shuffle=False,
                                 collate_fn=traffNet_collect_fn)
    actual_train_batch_size = train_dataloader.batch_size
    total_train_batches = math.ceil(len(train_dataset) / actual_train_batch_size)
    actual_val_batch_size = val_dataloader.batch_size
    total_val_batches = math.ceil(len(val_dataset) / actual_val_batch_size)
    actual_test_batch_size = test_dataloader.batch_size
    total_test_batches = math.ceil(len(test_dataset) / actual_test_batch_size)

    headers = ["", "batch_size", "batch_num"]
    data = [
        ["Train", actual_train_batch_size, total_train_batches],
        ["Val", actual_val_batch_size, total_val_batches],
        ["Test", actual_test_batch_size, total_test_batches]
    ]

    column_widths = [max(len(str(row[i])) for row in [headers] + data) for i in range(len(headers))]

    table_width = sum(column_widths) + len(headers) * 3
    logger.info("-" * table_width)
    logger.info("|".join(headers[i].center(column_widths[i] + 2) for i in range(len(headers))))
    logger.info("-" * table_width)
    for row in data:
        logger.info("|".join(str(item).center(column_widths[i] + 2) for i, item in enumerate(row)))
    logger.info("-" * table_width)
    logger.info("Load data success!")
    return train_dataloader, val_dataloader, test_dataloader
