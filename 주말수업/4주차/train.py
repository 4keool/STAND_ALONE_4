# 필요한 패키지를 임포트 
import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# 하이퍼파라메터 설정 
# optimizer에 사용 될 learning rate 를 선언 
lr = 0.001
# MNIST 이미지 크기 (width, height )
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 500 
total_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# 학습을 할까요? 학습 Loop 설정 (for / while)
for epoch in range(total_epochs): 
    # 데이터 로더가 데이터를 넘겨주기 
    for idx, (image, label) in enumerate(train_loader) : 
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
        # 중간 성능 평가 

