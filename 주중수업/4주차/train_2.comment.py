# 필요한 패키지를 임포트 
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

# 하이퍼파라메터 설정 
# cuda가 사용 유무에 따라 할당하는 디바이스를 지정함
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DB에서 사용되는 이미지의 w,h
image_size = (28, 28)
# hidden layer의 크기
hidden_size = 500
# 분류하고자 하는 클래스의 수
num_classes = 10
# 네트워크에 들어가는 데이터의 사이즈 크기
batch_size = 100
# optimizer에 사용될 learning rate
lr = 0.001
# 전체 데이터셋이 학습되는 횟수
total_epochs = 3

# 모델 class (설계도) 만들기 
class MLP(nn.Module):
    # 클래스 초기화
    def __init__(self, image_size, hidden_size, num_classes):
        # 부모 클래스의 메소드를 초기화 
        super().__init__()
        
        # image_size를 저장
        self.image_size = image_size

        # mlp 네트워크 생성
        self.mlp1 = nn.Linear(in_features=image_size[0]*image_size[1], out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)

    # 실제 연산이 일어나는 함수
    def forward(self, x):
        # x : [batch_size, 28, 28, 1] 
        # mlp1에 입력과 맞는 형태로 reshape 
        x = torch.reshape(x, (-1, self.image_size[0] * self.image_size[1]))
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

        if idx % 100 == 0:
            print(idx, loss)