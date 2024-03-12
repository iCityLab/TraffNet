from traffnet import TraffNet

if __name__ == '__main__':
    model = TraffNet(r'./configs/taxi_bj-1.yaml')
    model.train()
