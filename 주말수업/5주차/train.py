# 필요한 패키지를 임포트 
import os 
import json
import torch 
import argparse
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def parse_args() :
    parser = argparse.ArgumentParser()
    # 하이퍼파라메터 설정 -> parser 
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--total_epochs', type=int, default=3)
    parser.add_argument('--results_folder', type=str, default='results')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--do_save', action='store_true', help='if given, save results')
    parser.add_argument('--data', nargs='+', type=str)

    args = parser.parse_args()
    return args

def main() :
    args = parse_args()

    # 저장 상위 폴더 생성 
    if not os.path.exists(args.results_folder): 
        os.makedirs(args.results_folder)

    # 저장 폴더 생성 & 그 경로를 저장 
    # 폴더 이름 알아내고, 폴더 만들기 (상위 폴더 내부의 구조를 파악하고 이로부터 유추)
    folder_name = max([0] + [int(e) for e in os.listdir(args.results_folder)])+1
    save_path = os.path.join(args.results_folder, str(folder_name))
    os.makedirs(save_path)

    # Hparam 저장하기 
    with open(os.path.join(save_path, 'hparam.json'), 'w') as f :
        write_args = args.__dict__
        del write_args['device']
        json.dump(write_args, f, indent=4)

    assert() 
    # 모델 class (설계도) 만들기 
    class MLP(nn.Module): 
        def __init__(self, image_size, hidden_size, num_classes) : 
            # 상속 해주는 클래스를 부팅 
            super().__init__()
            
            self.image_size = image_size
            self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
            self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
            self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
        
        def forward(self, x) : 
            # x : [batch_size, 28, 28, 1] 
            batch_size = x.shape[0]
            # reshape 
            x = torch.reshape(x, (-1, self.image_size * self.image_size))
            # mlp1 ~ mlp4 진행 
            x = self.mlp1(x)
            x = self.mlp2(x)
            x = self.mlp3(x)
            x = self.mlp4(x)
            # 출력 
            return x

    # 모델 객체 만들기 <- 하이퍼파라미터 사용 
    myMLP = MLP(args.image_size, args.hidden_size, args.num_classes).to(args.device)

    # 데이터 불러오기 
    # 데이터셋 만들기 
    train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
    test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)

    # 데이터 로더 만들기 
    train_loader = DataLoader(dataset=train_mnist, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_mnist, batch_size=args.batch_size, shuffle=True)

    # Loss 함수 만들기 
    loss_fn = nn.CrossEntropyLoss()

    # optimizer 만들기 
    optim = Adam(params=myMLP.parameters(), lr=args.lr)

    def evaluate(model, testloader, device): 
        model.eval()
        total = 0 
        correct = 0 
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            output_index = torch.argmax(output, dim=1) # [7, 2, 8, 3, 0, ...]
            
            correct += (output_index == label).sum().item()
            total += label.shape[0]
        acc = correct / total * 100 
        model.train()
        return acc # scalar

    def evaluate_by_class(model, testloader, device, num_classes): 
        model.eval()
        total = torch.zeros(num_classes)
        correct = torch.zeros(num_classes)
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            output_index = torch.argmax(output, dim=1) # [7, 2, 8, 3, 0, ...]
            
            for idx in range(num_classes): 
                total[idx] += (label == idx).sum().item()
                correct[idx] += ((label == idx) * (output_index == idx)).sum().item()
        acc = correct / total
        model.train()
        return acc # vector 

    _max = -1
    # 학습을 할까요? 학습 Loop 설정 (for / while)
    for epoch in range(args.total_epochs): 
        # 데이터 로더가 데이터를 넘겨주기 
        for idx, (image, label) in enumerate(train_loader) : 
            image = image.to(args.device)
            label = label.to(args.device)

            # 모델이 추론 
            output = myMLP(image)

            # 출력물을 바탕으로 loss 계산 
            loss = loss_fn(output, label)

            # 파라미터 업데이트 (Optimizer)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if idx % 100 == 0 : 
                print(loss)
                # 중간 성능 평가 
                acc = evaluate(myMLP, test_loader, args.device)
                # acc_class = evaluate_by_class(myMLP, test_loader, device, num_classes)
                # 성능이 제일 좋을 때, 모델 저장 
                # 이전 최고값(_max) 보다 지금의 acc가 더 수치가 높다 -> 성능이 제일 좋다 
                if _max < acc :
                    print('새로운 max값 달성. 모델 저장 ', acc)
                    _max = acc 
                    # 저장하셈 
                    torch.save(
                        myMLP.state_dict(),
                        os.path.join(save_path, 'myMLP_best.ckpt')
                    )

if __name__ == '__main__' : 
    main() 