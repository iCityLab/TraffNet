import logging
import os
import sys
import time

import torch
import yaml
from tqdm import tqdm

from traffnet.utils.data_utils import process_ps_dataset, process_fp_dataset
from traffnet.utils.model_utils import parse_model
from traffnet.utils.traffNet_utils import od2pathNum


class TraffNet:
    def __init__(self, config_path):
        super(TraffNet, self).__init__()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("TraffNet")
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.logger.info("load config: %s", self.config["name"])
            self.logger.info("Model start loaded")
        except Exception as e:
            self.logger.error("Failed to load config file. Error: %s", e)
            sys.exit()

        self.network = self.config["network"]
        self.task = self.config["task"]
        if self.task == "path_select":
            self.network_seq = self.network["path_select"]
        elif self.task == "flow_predict":
            self.network_seq = self.network["path_select"]
            self.network_seq.extend(self.network["flow_predict"])
        else:
            self.logger.error("Task error")
            sys.exit()
        self.models, self.save_model = parse_model(self.logger, self.network_seq, self.config)

        if self.task == "flow_predict" and self.config["path_select_pkl"]:
            # 加载预训练模型的参数
            try:
                model = torch.load(self.config["path_select_pkl"])
                pkl_directory = os.path.dirname(self.config["path_select_pkl"])
                state_dict_path = os.path.join(pkl_directory, "pretrain_path_select.pth")
                torch.save(model.state_dict(), state_dict_path)
                pre_weights_dict = torch.load(state_dict_path, map_location=torch.device('cuda:0'))
                missing_keys, unexpected_keys = self.models.load_state_dict(pre_weights_dict, strict=False)
                self.logger.info(f'missing_key:{missing_keys}')
                self.logger.info(f'unexpected_key:{unexpected_keys}')

                self.logger.info("Path select model state loaded")
            except TypeError as e:
                self.logger.error("Failed to load model state. Error: %s", e)
                sys.exit()

    def train(self):
        device = torch.device(self.config["device"])
        if self.task == "path_select":
            train_dataloader, val_dataloader, _ = process_ps_dataset(self.logger, self.config)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.models.parameters(), lr=0.0001)
            epochs = self.config["train"]["path_select"]["epochs"]
        else:
            train_dataloader, val_dataloader, _ = process_fp_dataset(self.logger, self.config)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.models.parameters(), lr=0.0001)
            epochs = self.config["train"]["flow_predict"]["epochs"]

        min_val_total_loss = 100000000
        result_index = 0
        while (os.path.exists(f'./results/{self.config["task"]}/{self.config["name"]}({result_index})')):
            result_index += 1
        result_path = f'./results/{self.config["task"]}/{self.config["name"]}({result_index})'
        os.makedirs(result_path)
        for epoch in range(epochs):
            epoch_start_time = time.time()
            train_total_loss = 0
            self.models.train()
            for data in tqdm(train_dataloader, desc=f'Train epoch {epoch}/{epochs}: '):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                batchGraphSeq = data[0].to(device)
                labels = data[1].to(torch.float32).to(device)
                logits = self.models(batchGraphSeq)
                if self.task == "path_select":
                    g_select = logits.edge_type_subgraph(['select-', 'select+'])
                    logits = od2pathNum(g_select, False)
                loss = criterion(logits, labels)
                train_total_loss = train_total_loss + loss.item()
                loss.backward()
                optimizer.step()

            self.models.eval()
            val_total_loss = 0
            with torch.no_grad():
                for val_data in tqdm(val_dataloader, desc=f'Val epoch {epoch}/{epochs}: '):
                    batchGraphSeq = val_data[0].to(device)
                    labels = val_data[1].to(torch.float32).to(device)
                    logits = self.models(batchGraphSeq)
                    if self.task == "path_select":
                        g_select = logits.edge_type_subgraph(['select-', 'select+'])
                        logits = od2pathNum(g_select, False)
                    loss = criterion(logits, labels)
                    val_total_loss = val_total_loss + loss.item()

            train_loss = train_total_loss / len(train_dataloader)
            val_loss = val_total_loss / len(val_dataloader)
            epoch_end_time = time.time()

            with open(f'{result_path}/loss.txt', 'a') as f:
                f.write(
                    f"[epoch:{epoch} | train_total_loss:{train_total_loss},val_total_loss:{val_total_loss} | avgbatchTrainLoss:{train_loss},avgbatchValLoss:{val_loss} | time:{epoch_end_time - epoch_start_time}" + '\n')
            self.logger.info(
                f"[epoch:{epoch} | train_total_loss:{train_total_loss},val_total_loss:{val_total_loss} | avgbatchTrainLoss:{train_loss},avgbatchValLoss:{val_loss} | time:{epoch_end_time - epoch_start_time}")

            if min_val_total_loss > val_total_loss:
                torch.save(self.models, f'{result_path}/model.pkl')
                min_val_total_loss = val_total_loss

    def test(self):
        """
        TODO: TraffNet测试集验证功能
        """
        pass

    def predict(self):
        """
        TODO: TraffNet预测功能
        """
        pass
