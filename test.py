from models import AutoEncoder
import torch
from torch import nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load a single image from file
image_path = 'capy.jpg'

image = Image.open(image_path).convert('RGB')
image = preprocess(image)

input = image.unsqueeze(0)

model = AutoEncoder(128, 64)

output = model(input)

# print(input)
# print(output.shape)
# print(output)
loss_fn = nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)


model.train()
for epoch in tqdm(range(50000)):

    output = model(input)

    loss = loss_fn(output, input)
    if epoch % 1000 == 0:
        print(epoch)
        print(loss)
    loss.backward()

    optim.step()
    optim.zero_grad()

model.eval()
with torch.no_grad():
    output = model(input)

image1_np = input.squeeze().permute(1, 2, 0).numpy()
image2_np = output.squeeze().permute(1, 2, 0).numpy()

# Display the images using matplotlib
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image1_np)
axs[0].axis('off')
axs[0].set_title('Input')

axs[1].imshow(image2_np)
axs[1].axis('off')
axs[1].set_title('Output')

plt.show()