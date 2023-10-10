import torch
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import to_pil_image
from model_pycharm import UNet
from data import TerahertzDataset, train_val_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import pandas as pd
import matplotlib.pyplot as plt

# Define the evaluation metrics
def calculate_psnr(output: torch.Tensor, target: torch.Tensor) -> float:
    mse = mse_loss(output, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_mse(output: torch.Tensor, target: torch.Tensor) -> Tensor:
    return mse_loss(output, target)

def calculate_mae(output: torch.Tensor, target: torch.Tensor) -> Tensor:
    return nn.functional.l1_loss(output, target)

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
model.load_state_dict(torch.load('model_2.pth'))
model.to(device)
data_transform = Compose([ToTensor()])
# Load the trained model weights here
dataset = TerahertzDataset(root_dir='D:\THz\enhancement_pytorch', transform=data_transform)
#Split the dataset into training, validation, and testing sets
_, val_indices, _ = train_val_test_split(dataset, train_percent=.6, val_percent=.2,
                                                                test_percent=.2, shuffle=False)

val_dataset = torch.utils.data.Subset(dataset, val_indices)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
psnr_list = []
mse_list = []
mae_list = []

with torch.no_grad():
    for i, (lr_images, hr_images) in enumerate(val_dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        outputs = model(lr_images)
        psnr = calculate_psnr(outputs, hr_images)
        psnr_list.append(psnr)
        mse = calculate_mse(outputs, hr_images)
        mse_list.append(mse)
        mae = calculate_mae(outputs, hr_images)
        mae_list.append(mae)
        print(f'Image {i + 1} PSNR: {psnr:.4f} MSE: {mse:.4f} MAE: {mae:.4f}')

# mse_list = [tensor.float() for tensor in mse_list]
# mae_list = [tensor.float() for tensor in mae_list]

# Convert the list of tensors to a single tensor object
psnr_tensor = torch.stack(psnr_list)
mse_tensor = torch.stack(mse_list)
mae_tensor = torch.stack(mae_list)

# Move the tensor objects to CPU memory
psnr_tensor = psnr_tensor.cpu()
mse_tensor = mse_tensor.cpu()
mae_tensor = mae_tensor.cpu()

# Initialize DataFrame
loss_df = pd.DataFrame({'psnr': psnr_tensor, 'mse': mse_tensor, 'mae': mae_tensor})

# Save to Excel file
loss_df.to_excel('psnr_mse_mae_1.xlsx', index=False)

plt.plot(psnr_tensor)
plt.xlabel('Image')
plt.ylabel('PSNR')
plt.title('Peak Signal-to-Noise Ratio (PSNR)')
plt.show()

plt.plot(mse_tensor)
plt.xlabel('Image')
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE)')
plt.show()

plt.plot(mae_tensor)
plt.xlabel('Image')
plt.ylabel('MAE')
plt.title('Mean Absolute Error (MAE)')
plt.show()
