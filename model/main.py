import argparse
import platform
from dataclasses import dataclass
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

PYTHON_VERSION = platform.python_version()


# TODO: Abstract to config package with env detection
@dataclass
class Config:
    """Configuration class for hyperparameters and settings."""

    LEARNING_RATE = 0.1
    EPOCHS = 10
    BATCH_SIZE = 64
    PATIENCE = 3
    MIN_DELTA = 0.01
    DATA_DIR = "./data"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLflow settings
    EXPERIMENT_NAME = "mnist-cnn-experiments"
    MLFLOW_TRACKING_URI = "file:./mlruns"  # Can be changed to remote server


class MNISTCNN(nn.Module):
    """Convolutional Neural Network for MNIST digit classification."""

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, num_classes, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


class EarlyStopper:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class MLflowMNISTTrainer:
    """MNIST trainer with MLflow integration."""

    def __init__(self, config=None, run_name=None, experiment_tags=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.run_name = run_name
        self.experiment_tags = experiment_tags or {}

        # MLflow setup
        self.setup_mlflow()

        # Tracking variables
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")

    def setup_mlflow(self):
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.config.EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.config.EXPERIMENT_NAME,
                    tags={"purpose": "MNIST CNN training experiments"},
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not set up experiment: {e}")
            experiment_id = None

        self.experiment_id = experiment_id
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {self.config.EXPERIMENT_NAME}")

    def setup_data(self):
        """Setup data loaders for training, validation, and testing."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load datasets
        full_train_ds = datasets.MNIST(
            root=self.config.DATA_DIR, train=True, download=True, transform=transform
        )

        test_ds = datasets.MNIST(
            root=self.config.DATA_DIR, train=False, download=True, transform=transform
        )

        # Split training into train/validation
        train_ds, val_ds = random_split(full_train_ds, [50000, 10000])

        # Create data loaders
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Log dataset info
        mlflow.log_param("train_samples", len(train_ds))
        mlflow.log_param("val_samples", len(val_ds))
        mlflow.log_param("test_samples", len(test_ds))

        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")

    def setup_model(self, dropout_rate=0.5):
        """Initialize model and optimizer."""
        self.model = MNISTCNN(dropout_rate=dropout_rate).to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=1e-4,
        )

        # Log model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        mlflow.log_param("total_parameters", total_params)
        mlflow.log_param("trainable_parameters", trainable_params)
        mlflow.log_param("dropout_rate", dropout_rate)

        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += target.size(0)

            # Log batch metrics every 100 batches
            if batch_idx % 100 == 0:
                mlflow.log_metric(
                    "batch_loss",
                    loss.item(),
                    step=epoch * len(self.train_loader) + batch_idx,
                )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total_samples

        return avg_loss, accuracy

    def validate(self, epoch):
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

        # Track best metrics
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss

        return avg_loss, accuracy

    def train(self):
        """Main training loop with MLflow tracking."""
        early_stopper = EarlyStopper(
            patience=self.config.PATIENCE, min_delta=self.config.MIN_DELTA
        )

        print(f"\nStarting training for {self.config.EPOCHS} epochs...")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
        print("-" * 55)

        for epoch in range(self.config.EPOCHS):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            print(
                f"{epoch + 1:5d} | {train_loss:10.4f} | {train_acc:9.2f}% | {val_loss:8.4f} | {val_acc:7.2f}%"
            )

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            # Save model checkpoint if it's the best so far
            if val_acc > self.best_val_acc or val_loss < self.best_val_loss:
                self.save_checkpoint(epoch, val_acc, val_loss, is_best=True)

            if early_stopper(val_loss, epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("early_stop_epoch", epoch + 1)
                break
        else:
            mlflow.log_param("early_stopped", False)

        # Log final best metrics
        mlflow.log_metric("final_best_val_accuracy", self.best_val_acc)
        mlflow.log_metric("final_best_val_loss", self.best_val_loss)

        print("Training completed!")

    def evaluate(self):
        """Evaluate model on test set."""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += pred[i].eq(target[i]).item()
                    class_total[label] += 1

        accuracy = 100.0 * correct / total

        # Log test metrics
        mlflow.log_metric("test_accuracy", accuracy)

        # Log per-class accuracy
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                mlflow.log_metric(f"test_accuracy_class_{i}", class_acc)

        print(f"\nTest Accuracy: {accuracy:.2f}%")

        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                class_acc = 100.0 * class_correct[i] / class_total[i]
                print(f"Class {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")

        return accuracy

    def save_checkpoint(self, epoch, val_acc, val_loss, is_best=False):
        """Save model checkpoint and log to MLflow."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_best:
            filename = f"best_model_acc_{val_acc:.2f}_epoch_{epoch + 1}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch + 1}_{timestamp}.pth"

        # Save model state
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "config": self.config.__dict__,
        }

        # Save locally
        torch.save(checkpoint, filename)

        # Log model to MLflow
        if is_best:
            mlflow.pytorch.log_model(self.model, "best_model", extra_files=[filename])
            mlflow.log_artifact(filename, "checkpoints")

        print(f"Model saved: {filename}")

    def log_system_info(self):
        """Log system and environment information."""

        mlflow.log_param("python_version", PYTHON_VERSION)
        mlflow.log_param("pytorch_version", torch.__version__)
        mlflow.log_param("device", str(self.device))
        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param(
                "gpu_memory",
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
            )

    def run_experiment(self, dropout_rate=0.5):
        """Run complete experiment with MLflow tracking."""
        with mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id):
            # Log hyperparameters
            mlflow.log_params(
                {
                    "learning_rate": self.config.LEARNING_RATE,
                    "batch_size": self.config.BATCH_SIZE,
                    "epochs": self.config.EPOCHS,
                    "patience": self.config.PATIENCE,
                    "min_delta": self.config.MIN_DELTA,
                    "optimizer": "SGD",
                    "criterion": "CrossEntropyLoss",
                }
            )

            # Log experiment tags
            for key, value in self.experiment_tags.items():
                mlflow.set_tag(key, value)

            # Log system info
            self.log_system_info()

            # Setup and train
            self.setup_data()
            self.setup_model(dropout_rate)
            self.train()
            test_accuracy = self.evaluate()

            # Log final model
            mlflow.pytorch.log_model(self.model, "final_model")

            print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
            print(
                f"MLflow Run URL: {mlflow.get_tracking_uri()}/#/experiments/{self.experiment_id}/runs/{mlflow.active_run().info.run_id}"
            )

            return test_accuracy


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MNIST CNN with MLflow")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--run-name", type=str, help="Name for this MLflow run")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="mnist-cnn-experiments",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-uri", type=str, default="file:./mlruns", help="MLflow tracking URI"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Update config with command line arguments
    config = Config()
    config.LEARNING_RATE = args.lr
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.PATIENCE = args.patience
    config.EXPERIMENT_NAME = args.experiment_name
    config.MLFLOW_TRACKING_URI = args.mlflow_uri

    print("MNIST CNN Training with MLflow")
    print("=" * 50)
    print(f"Device: {config.DEVICE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Dropout: {args.dropout}")

    # Setup experiment tags
    experiment_tags = {
        "model_type": "CNN",
        "dataset": "MNIST",
        "framework": "PyTorch",
        "task": "classification",
    }

    # Initialize trainer
    trainer = MLflowMNISTTrainer(
        config=config, run_name=args.run_name, experiment_tags=experiment_tags
    )

    # Run experiment
    accuracy = trainer.run_experiment(dropout_rate=args.dropout)

    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
