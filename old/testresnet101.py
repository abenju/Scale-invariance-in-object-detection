import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from models.resnet101 import resnet101
#%%

img_path = "../img/test/samoyed2.jpg"

#open image
img = Image.open(img_path)

with open("../labels/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

#%%
# prepare image instructions
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#%%
# prepare image instructions
preprocess = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#%%
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

#%%
plt.imshow(input_tensor.permute(1, 2, 0))
plt.show()

#%%
model = resnet101(pretrained=True)
model.eval()

#%%
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

#%%
with torch.no_grad():
    output = model(input_batch)

#%%
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

#%%
top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
