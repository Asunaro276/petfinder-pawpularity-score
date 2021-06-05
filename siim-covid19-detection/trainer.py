import torch
from tqdm.notebook import tqdm


class Trainer:
    def __init__(self, train_dataloader, val_dataloader, model, optimizer, criterion):
        self.train = train_dataloader
        self.valid = val_dataloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_cycle(self):
        self.model.train()
        train_prog_bar = tqdm(self.train, total=len(self.train))
        epoch_loss = 0.0
        epoch_corrects = 0
        for images, labels in train_prog_bar:
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_corrects += torch.sum(preds == labels.data)

        epoch_loss = epoch_loss / len(self.train.dataset)
        epoch_acc = epoch_corrects.double() / len(self.train.dataset)

        return epoch_loss, epoch_acc

    def valid_one_cycle(self):
        self.model.eval()

        valid_prog_bar = tqdm(self.valid, total=len(self.valid))

        epoch_loss = 0.0
        epoch_corrects = 0
        with torch.no_grad():
            for images, labels in valid_prog_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                epoch_loss += loss.item() * images.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(self.valid.dataset)
            epoch_acc = epoch_corrects.double() / len(self.valid.dataset)

        return epoch_loss, epoch_acc

