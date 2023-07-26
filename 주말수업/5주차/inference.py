# 패키지 임포트
import os 
import torch 
import torch.nn as nn
from PIL import Image 
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

# 타겟하는 학습 세팅을 설정 
target_folder = '../../주중수업/5주차/results/6'
assert os.path.exists(target_folder), 'target folder doesnt exists'

# 하이퍼파라메터 로드 
with open(os.path.join(target_folder, 'hparam.txt'), 'r') as f: 
    data = f.readlines()

lr = float(data[0].strip())
image_size = int(data[1].strip())
num_classes = int(data[2].strip())
batch_size = int(data[3].strip())
hidden_size = int(data[4].strip())
total_epochs = int(data[5].strip())
results_folder = data[6].strip()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 class 만들기 
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

# 모델 객체 만들기 
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

# 모델 weight 업데이트 
ckpt = torch.load(
        os.path.join(
            target_folder, 'myMLP_best.ckpt'
            )
        )
myMLP.load_state_dict(ckpt)


# 추론 데이터 가지고오기 
image_path = './test_image.jpg'
assert os.path.exists(image_path), 'target image doesnt exists'
input_image = Image.open(image_path).convert('L')

# 학습시 사용했던 전처리 과정을 그대로 실행 
# 크기 맞추기 -> tensor로 만들기 
resizer = Resize(image_size)
totensor = ToTensor() 
image = totensor(resizer(input_image)).to(device)

# 모델 추론 
output = myMLP(image)
# 추론 결과를 이해할 수 있는 형태로 변환 
output = torch.argmax(output).item()

print(f'Model says, the image is {output}')