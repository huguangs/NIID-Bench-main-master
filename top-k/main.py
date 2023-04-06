import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义客户端类
class Client:
    def __init__(self, model, train_loader, test_loader, lr=0.1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, accuracy

    def get_topk_grads(self, k=0.01):
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                importance = param.grad.abs().sum()
                k_ = int(importance.numel() * k)
                topk_values, _ = torch.topk(param.grad.abs().view(-1), k_)
                mask = torch.zeros_like(param.grad)
                mask[param.grad.abs() >= topk_values[-1]] = 1
                grads.append(mask * param.grad)
        return torch.cat(grads)

# 定义服务器类
class Server:
    def __init__(self, clients):
        self.clients = clients

    def aggregate(self):
        grads = None
        num_grads = 0
        for client in self.clients:
            client_grads = client.get_topk_grads()
            if grads is None:
                grads = client_grads
            else:
                grads += client_grads
            num_grads += 1
        grads /= num_grads
        return grads

# 定义主函数
def main():
    # 加载MNIST数据集
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    train_loaders = [torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) for _ in range(10)]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # 初始化客户端和服务器
    clients = [Client(nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)), train_loader, test_loader) for train_loader in train_loaders]
    server = Server(clients)

    # 训练和聚合
    for round in range(num_rounds):
        print(f"\nRound {round + 1}")

        # 客户端训练并计算梯度
        grad_list = []
        for client in clients:
            client.train()
            client_optimizer.zero_grad()
            # 获取客户端的本地数据
            inputs, labels = client.get_data()
            # 前向传播
            outputs = client.model(inputs)
            # 计算局部loss
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 获取梯度并压缩
            grad = client.get_topk_grads()
            grad_list.append(grad)

        # 服务器聚合梯度并更新模型
        server.aggregate(grad_list)
        server.update_model()

        # 客户端更新模型
        for client in clients:
            client.update_model(server.model)

        # 计算并打印全局loss
        global_loss = evaluate_global_loss(server.model, test_loader, criterion)
        print(f"Global Loss: {global_loss:.4f}")

