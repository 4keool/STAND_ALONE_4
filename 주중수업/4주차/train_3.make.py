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
img_size = (28, 28)
# hidden layer의 크기
hidden_size = 500
# 분류하고자 하는 클래스의 수
nClass = 10
# 네트워크에 들어가는 데이터의 사이즈 크기
batch_size = 100
# optimizer에 사용될 learning rate
lr = 0.001
# 전체 데이터셋이 학습되는 횟수
nEpoch = 3

# 모델 class (설계도) 만들기 
class MLP(nn.Module):
    # 클래스 초기화
    def __init__(self, imgSize, hidSize, num_Classes):
        # 부모 클래스의 메소드를 초기화
        super().__init__() 
        # image_size를 저장
        self.img_size = imgSize
        # mlp 네트워크 생성
        self.mlp1 = nn.Linear(self.img_size[0]*self.img_size[1], hidSize)
        self.mlp2 = nn.Linear(hidSize, hidSize)
        self.mlp3 = nn.Linear(hidSize, hidSize)
        self.mlp4 = nn.Linear(hidSize, num_Classes)

    # 실제 연산이 일어나는 함수
    def forward(self, x):
        # x : [batch_size, 28, 28, 1] 
        # mlp1에 입력과 맞는 형태로 reshape 
        x = torch.reshape(x, (-1, self.img_size[0] * self.img_size[1]))
        # mlp1 ~ mlp4 진행 
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        # 출력
        return x

# 모델 객체 만들기 <- 하이퍼파라미터 사용
myMLP = MLP(img_size, hidden_size, nClass).to(device)

# 데이터 불러오기 
# 데이터셋 만들기 
train_dataset = MNIST("../../data/",train=True, transform=ToTensor(), download=True)
test_dataset = MNIST("../../data/", train=False, transform=ToTensor(), download=False)

# 데이터 로더 만들기 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Loss 함수 만들기 
loss_func = nn.CrossEntropyLoss()

# optimizer 만들기 
opt = Adam(params=myMLP.parameters(), lr=lr)

# 학습을 할까요? 학습 Loop 설정 (for / while)
for n in range(nEpoch):
    # 데이터 로더가 데이터를 넘겨주기 
    for idx, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        # 모델이 추론 
        output = myMLP(img)
        # 출력물을 바탕으로 loss 계산
        loss = loss_func(output, label) 
        # 파라미터 업데이트 (Optimizer)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if idx%100==0:
            print(idx, loss)