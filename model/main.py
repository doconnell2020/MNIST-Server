import pathlib
from datetime import datetime

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse


class Config:
    """Configuration class for hyperparameters and settings."""
    LEARNING_RATE = 0.1
    EPOCHS = 10
    BATCH_SIZE = 64
    PATIENCE = 3
    MIN_DELTA = 0.01
    DATA_DIR = "./data"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MNISTCNN(nn.Module):
    """Convolutional Neural Network for MNIST digit classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, num_classes, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # More flexible than fixed pool size
        return x.view(x.size(0), -1)


class EarlyStopper:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MNISTTrainer:
    """Main training class that encapsulates the training process."""

    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def setup_data(self):
        """Setup data loaders for training, validation, and testing."""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])

        # Load datasets
        full_train_ds = datasets.MNIST(
            root=self.config.DATA_DIR,
            train=True,
            download=True,
            transform=transform
        )

        test_ds = datasets.MNIST(
            root=self.config.DATA_DIR,
            train=False,
            download=True,
            transform=transform
        )

        # Split training into train/validation
        train_ds, val_ds = random_split(full_train_ds, [50000, 10000])

        # Create data loaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")

    def setup_model(self):
        """Initialize model and optimizer."""
        self.model = MNISTCNN().to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=0.9,  # Add momentum for better convergence
            weight_decay=1e-4  # Add regularization
        )

        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self):
        """Main training loop."""
        early_stopper = EarlyStopper(
            patience=self.config.PATIENCE,
            min_delta=self.config.MIN_DELTA
        )

        print(f"\nStarting training for {self.config.EPOCHS} epochs...")
        print("Epoch | Train Loss | Val Loss | Val Acc")
        print("-" * 40)

        for epoch in range(self.config.EPOCHS):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"{epoch + 1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_acc:7.2f}%")

            if early_stopper(val_loss):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("Training completed!")

    def evaluate(self):
        """Evaluate model on test set."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        return accuracy

    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        print(f"Model saved to {filepath}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MNIST CNN')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--save-model', type=str, help='Path to save the model')
    return parser.parse_args()


def main():
    """Main execution function."""

    # Update config with command line arguments
    config = Config()

    print("MNIST CNN Training")
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")

    # Initialize trainer
    trainer = MNISTTrainer(config)

    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()

    # Train and evaluate
    trainer.train()
    accuracy = trainer.evaluate()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer.save_model(pathlib.Path(f"./models/mnist_cnn_acc_{accuracy}_{stamp}.pth") )


if __name__ == "__main__":
    main()