# 필요한 패키지를 임포트 
import os 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# 하이퍼파라메터 설정 
lr = 0.001
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 1000 
total_epochs = 3
# 저장 폴더 이름 정화기
results_folder = 'results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 저장 상위 폴더 생성 
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# 저장 폴더 생성 & 그 경로를 저장 
# 폴더 이름 알아내고, 폴더 만들기 (상위 폴더 내부의 구조를 파악하고 이로부터 유추)
folder_name = max([0] + [int(e) for e in os.listdir(results_folder)]) + 1
save_path = os.path.join(results_folder, str(folder_name))
os.makedirs(save_path)

# Hparam 저장하기 
with open(os.path.join(save_path, 'hparam.txt'), 'w') as f:
    f.write(f'{lr}\n')
    f.write(f'{image_size}\n')
    f.write(f'{num_classes}\n')
    f.write(f'{batch_size}\n')
    f.write(f'{total_epochs}\n')
    f.write(f'{results_folder}\n')

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
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

# 데이터 불러오기 
# 데이터셋 만들기 
train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)

# 데이터 로더 만들기 
train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=True)

# Loss 함수 만들기 
loss_fn = nn.CrossEntropyLoss()

# optimizer 만들기 
optim = Adam(params=myMLP.parameters(), lr=lr)

# 성능평가 함수 만들기
def evaluate(model, testloader, device):
    # 평가 모드로 변경
    model.eval()
    total = 0
    correct = 0
    # 로더로 이미지 및 라벨 받아오기
    for image, label in testloader:
        image, label = image.to(device), label.to(device)
        # 추론하기
        output = model(image)
        # 가장 높은 확률의 인덱스 찾기
        output_index = torch.argmax(output, dim=1)

        # 인덱스와 라벨을 비교하여 correct 증가시키기
        correct += (output_index == label).sum().item()
        # total 값 구하기
        total += label.shape[0]

    # 확률 구하기
    acc = correct / total * 100
    model.train()
    return acc

# 클래스별 성능평가 함수 만들기
def evaluate_by_class(model, testloader, device, num_classes):
    # 평가 모드로 변경
    model.eval()

    # class 개수만큼 total 및 correct 만들기
    total = torch.zeros(num_classes)
    correct = torch.zeros(num_classes)

    # 로더로 이미지 및 라벨 받아오기
    for image, label in testloader:
        image, label = image.to(device), label.to(device)

        # 추론하기
        output = model(image)

        # 가장 높은 확률의 인덱스 찾기
        output_index = torch.argmax(output, dim=1)

        # coorect와 total 값 구하기
        for idx in range(num_classes):
            total[idx] += (label == idx).sum().item()
            correct[idx] += ((label == idx) * (output_index == idx)).sum().item()

    # 확률 구하기
    acc = correct / total
    model.train()
    return acc

# 성능평가에서 나올 수 없는 숫자로 max 값을 생성
_max = -1
# 학습을 할까요? 학습 Loop 설정 (for / while)
for epoch in range(total_epochs):
    # 데이터 로더가 데이터를 넘겨주기
    for idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)

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

            #중간 성능 평가
            acc = evaluate(myMLP, test_loader, device)
            # acc_class = evaluate_by_class(myMLP, test_loader, device, num_classes)
            # 성능이 제일 좋을 때, 모델 저장
            # 이전 최고값(_max) 보다 지금의 acc가 더 수치가 높다 -> 성능이 제일 좋다
            if _max < acc:
                print('새로운 max값 달성, 모델 저장 ', acc)
                _max = acc
                # 저장
                torch.save(
                    # 모델의 state_dict를 저장
                    myMLP.state_dict(),
                    os.path.join(save_path, 'myMLP_best.ckpt')
                )