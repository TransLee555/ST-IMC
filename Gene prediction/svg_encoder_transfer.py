import timm
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import KFold
from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainnorm import get_normalizer
from PIL import Image
import os

# Automatically set the root directory to the current working directory
root = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory

# Set device for GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f"Using device: {device}")

# Load pre-trained tile encoder model from HuggingFace Hub
tile_encoder = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True,
                                 pretrained_cfg_overlay=dict(file=os.path.join(root, 'pretrained/pytorch_model.bin')))

# Freeze all layers of the model
for param in tile_encoder.parameters():
    param.requires_grad = False

# Unfreeze last few layers for fine-tuning
for param in tile_encoder.blocks[-3:].parameters():
    param.requires_grad = True
tile_encoder.norm.requires_grad = True
tile_encoder.fc_norm.requires_grad = True
tile_encoder.head_drop.requires_grad = True
tile_encoder.head.requires_grad = True

# Define a custom encoder class with a new classification head
class Encoder(nn.Module):
    def __init__(self, tile_encoder, output):
        super(Encoder, self).__init__()
        self.tile_encoder = tile_encoder
        self.head_drop = nn.Dropout(p=0.5)  # Dropout for regularization
        self.new_head = nn.Linear(1536, output)  # New output layer for classification

    def forward(self, x):
        x = self.tile_encoder(x)
        x = self.head_drop(x)
        x = self.new_head(x)
        return x

    def compute_loss(self, regression_output, target):
        # Compute MSE loss for regression tasks
        return F.mse_loss(regression_output.view(-1), target.view(-1))

# Stain normalization setup using the Vahadane method
target_image = stain_norm_target()
stain_normalizer = get_normalizer("vahadane")
stain_normalizer.fit(target_image)

# Image transformation pipeline for the model input
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Standard normalization for pre-trained models
])

def stain_norm_func(img):
    """
    Apply stain normalization and transformations to the image.
    """
    img = np.array(img)
    if img.var() > 200:  # Check if image variance is large enough for stain normalization
        try:
            img = stain_normalizer.transform(img)
        except:
            pass
    img = Image.fromarray(img)
    img = transform(img)
    return img

# Custom Dataset class for loading image patches and their corresponding labels
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.y = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "patch", self.y.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.y.iloc[idx, 1:].astype(np.float32).values, dtype=torch.float32)
        return image, label

    def get_filename(self, idx):
        return self.y.iloc[idx, 0]

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        """
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop_training = True

        return self.stop_training

# Main training loop
if __name__ == '__main__':
    # Construct relative paths
    csv_file = os.path.join(root, 'all_data.csv')

    # Prepare dataset and split into train, validation, and test sets
    image_dataset = ImageDataset(csv_file=csv_file, root_dir=root, transform=stain_norm_func)
    total_size = len(image_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_val_dataset, test_dataset = random_split(image_dataset, [train_size + val_size, test_size])

    # Save test set sample names for external validation
    test_sample_names = [image_dataset.get_filename(idx) for idx in test_dataset.indices]
    test_csv_path = os.path.join(root, 'external_validation_samples.csv')
    pd.DataFrame(test_sample_names, columns=["sample_name"]).to_csv(test_csv_path, index=False)

    # K-Fold Cross Validation setup
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Early stopping setup
    early_stopping = EarlyStopping(patience=5, delta=0.01)  # Set patience to 3 epochs

    # Training process for each fold
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_dataset)):
        print(f'Fold {fold + 1}')

        # Subsets for training and validation
        train_subset = Subset(train_val_dataset, train_indices)
        val_subset = Subset(train_val_dataset, val_indices)

        # DataLoader for batching the dataset
        train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=100, shuffle=False)

        # Initialize the model and optimizer
        model = Encoder(tile_encoder, output=138).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Start training
        num_epochs = 50
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = model.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = model.compute_loss(outputs, labels)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Check for early stopping
            if early_stopping.check_early_stop(val_loss):
                print("Early stopping triggered")
                break  # Stop training if early stopping is triggered

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(root, 'save_model',
                                         f'best_model_epoch.pth')
                torch.save(model.state_dict(), save_path)

    print('Training complete!')




