# %%
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# %%
def z_score_normalize(input_tensor):
    mean = torch.mean(input_tensor)  # 计算均值
    std = torch.std(input_tensor)  # 计算标准差
    normalized_tensor = (input_tensor - mean) / std  # 标准化
    return normalized_tensor

# %%
# 1. Load image
circle_image = Image.open('./data/circle2C/circle.png').convert('L').resize((592,592))
c_image = Image.open('./data/circle2C/C.png').convert('L').resize((592,592))
circle_image=torch.from_numpy(np.array(circle_image)).float().unsqueeze(0).unsqueeze(0)
c_image=torch.from_numpy(np.array(c_image)).float().unsqueeze(0).unsqueeze(0)
circle_image = z_score_normalize(circle_image)
c_image = z_score_normalize(c_image)
print("circle_image.shape: ", circle_image.shape)
print("c_image.shape: ", c_image.shape)

# %%
from register.SpatialTransformer2 import SpatialTransformer2
from register.LocalDisplacementEnergy import BendingEnergy

model=SpatialTransformer2(circle_image.shape[2:],10)  # 生成模型
device=torch.device("cuda:1")  # 使用GPU
circle_image=circle_image.to(device)
c_image=c_image.to(device)
model=model.to(device)
opt=torch.optim.Adam(model.parameters(), lr=0.05)  # Adam优化器
loss_fn=nn.MSELoss()   # MSE作为loss
bending_energy = BendingEnergy(alpha=1, dimension=2)  # 弯曲能量作为正则项

for i in range(20000):
    loss=loss_fn(model(circle_image), c_image)  # 计算loss
    flows=[flows for flows in model.parameters()][0]  # 获取流场
    loss += bending_energy(flows)  # 加上正则项

    if (i + 1) % 200 == 0:
        print(i, " loss(with reg): ", loss)  # 打印loss

    opt.zero_grad()  # 梯度清零
    loss.backward(retain_graph=True)  # 反向传播
    opt.step()  # 更新参数

# %%
for i in range(20000,60000):
    loss=loss_fn(model(circle_image), c_image)  # 计算loss
    flows=[flows for flows in model.parameters()][0]  # 获取流场
    loss += bending_energy(flows)  # 加上正则项

    if (i + 1) % 200 == 0:
        print(i, " loss(with reg): ", loss)  # 打印loss

    opt.zero_grad()  # 梯度清零
    loss.backward(retain_graph=True)  # 反向传播
    opt.step()  # 更新参数

# %%
torch.save(model.state_dict(), './data/circle2C/model/train_model_name.pth')  # 保存模型

# %%
loss_fn(model(circle_image), c_image),loss_fn(circle_image, c_image)  # 计算loss

# %%
import matplotlib.pyplot as plt
from register.SpatialTransformer2 import SpatialTransformer2
from register.LocalDisplacementEnergy import BendingEnergy

model=SpatialTransformer2(circle_image.shape[2:],10)  # 生成模型
model.load_state_dict(torch.load('./data/circle2C/model/best_model.pth'))  # 加载模型
device=torch.device("cuda:1")  # 使用GPU
circle_image=circle_image.to(device)
c_image=c_image.to(device)
model=model.to(device)
output=model(circle_image).cpu().detach().numpy().squeeze()
output=(output-np.min(output))/(np.max(output)-np.min(output))*255
image=Image.fromarray(output.astype(np.uint8))
image.save('./data/circle2C/output/output.png')
plt.imshow(output, cmap='gray')
plt.show()

# %%
lattice=np.ones((592,592))
lattice[0:592:25,:]=0
lattice[:,0:592:25]=0
lattice=lattice*255
norm_lattice = z_score_normalize(torch.from_numpy(np.array(lattice)).float().unsqueeze(0).unsqueeze(0))
warped_norm_lattice=model(norm_lattice.to(device)).squeeze().cpu().detach().numpy().squeeze()

plt.subplot(1,2,1)
plt.imshow(lattice,cmap='gray')

plt.subplot(1,2,2)
warped_norm_lattice=(warped_norm_lattice-np.min(warped_norm_lattice))/(np.max(warped_norm_lattice)-np.min(warped_norm_lattice))*255
image=Image.fromarray(warped_norm_lattice.astype(np.uint8))
image.save('./data/circle2C/output/warped_norm_lattice.png')
plt.imshow(warped_norm_lattice, cmap='gray')
plt.show()

# %% [markdown]
# inverse

# %%
from register.InverseSpatialTransformer2 import InverseSpatialTransformer2
Inverse_model=InverseSpatialTransformer2(circle_image.shape[2:],10)

device=torch.device("cuda:1")
Inverse_model.load_state_dict(torch.load('./data/circle2C/model/best_model.pth'))

Inverse_model=Inverse_model.to(device)

loss_fn=nn.MSELoss()

inverse=Inverse_model(c_image)
loss_fn(inverse,circle_image)



# %%
import matplotlib.pyplot as plt
inverse=inverse.cpu().detach().numpy().squeeze()
inverse=(inverse-np.min(inverse))/(np.max(inverse)-np.min(inverse))*255
image=Image.fromarray(inverse.astype(np.uint8))
image.save('./data/circle2C/output/inverse.png')
plt.imshow(inverse, cmap='gray')
plt.show()


