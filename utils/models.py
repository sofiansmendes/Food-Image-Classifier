import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from torchvision import models
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from .datahandlers import ImageDataset
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImageClassifier(object):
    
    def __init__(self, train_df, val_df, test_df, params, train_strat='feature_extraction'):
        
        self.params = params

        class_to_idx = {lab:val for val,lab in enumerate(train_df['choice'].unique())}
    
        # Load Data from folders
        self.datasets = {
            'train': ImageDataset(
                        image_paths=train_df['image'],
                        image_labels=train_df['choice'],
                        class_to_idx=class_to_idx,
                        data_obj='train'),
            'val': ImageDataset(
                        image_paths=val_df['image'],
                        image_labels=val_df['choice'],
                        class_to_idx=class_to_idx,
                        data_obj='val'),
            'test': ImageDataset(
                        image_paths=test_df['image'],
                        image_labels=test_df['choice'],
                        class_to_idx=class_to_idx,
                        data_obj='test')
        }

        self.num_classes = self.datasets['train'].num_classes()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.train_strat = train_strat
        # Freeze model parameters in case of 'feature_extraction' strategy
        if self.train_strat == 'feature_extraction':
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)

        self.model = self.model.to(device)
        
        # Since last layer is linear, it is possible to use the cross entropy loss directly.
        # If one wants to use NLLLoss, a logsoftmax activation haS to be used in the last layer.
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = getattr(optim, self.params['optimizer'])(self.model.parameters(), lr= self.params['lr'], weight_decay=self.params['L2'])
        
        # Initialize the AUC metric
        self.auc_eval = BinaryAUROC().to(device)

        # Initialize the Accuracy metric with 0.5 threshold
        self.acc_eval = BinaryAccuracy().to(device)
    
        # Decay LR by a factor of 0.1 every 7 epochs
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        
    def train(self, batch_size, num_epochs=25, model_save_path='best_train_model.pth'):
        since = time.time()
        
        # Batch size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.train_data_size = len(self.datasets['train'])
        self.val_data_size = len(self.datasets['val'])

        # Create iterators for the Data loaded using DataLoader module
        self.train_loader = DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=True)

        self.train_loss_values = []
        self.train_acc_values = []
        self.train_auc_values = []
        self.val_loss_values = []
        self.val_acc_values = []
        self.val_auc_values = []

        last_improvement = 0
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch+1, self.num_epochs))
            # Set to training mode
            self.model.train()
            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0
            train_auc = 0.0
            val_loss = 0.0
            val_acc = 0.0
            val_auc = 0.0

            # Start the batch training loop
            for i, (inputs, labels) in enumerate(self.train_loader):
                # Send inputs and labels to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Clean existing gradients
                self.optimizer.zero_grad()
                # Forward pass - compute outputs on input data using the model
                outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(outputs, labels)
                # Backpropagate the gradients
                loss.backward()
                # Update the parameters
                self.optimizer.step()
                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                # Applies Softmax to obtain the probabilities
                probs = nn.Softmax(dim=1)(outputs)[:,1]
                # Compute the AUC
                auc = self.auc_eval(probs, labels)
                # Compute the accuracy
                acc = self.acc_eval(probs, labels)
                # Compute total auc and accuracy in the whole batch and add to train auc and train_acc
                train_auc += auc.item() * inputs.size(0)
                train_acc += acc.item() * inputs.size(0)
                print("Batch: {:02d}, Train Loss: {:.4f}, Batch Train AUC {:.4f}, Acc: {:.4f}".format(i, loss.item(), auc.item(), acc.item()))

            # Validation - No gradient tracking needed
            with torch.no_grad():
                # Set to evaluation mode
                self.model.eval()
                # Validation loop
                for j, (inputs, labels) in enumerate(self.val_loader):
                    # Send inputs and labels to GPU
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Forward pass - compute outputs on input data using the model
                    outputs = self.model(inputs)
                    # Compute loss
                    loss = self.criterion(outputs, labels)
                    # Compute the total loss for the batch and add it to valid_loss
                    val_loss += loss.item() * inputs.size(0)
                    # Applies Softmax to obtain the probabilities
                    probs = nn.Softmax(dim=1)(outputs)[:,1]
                    # Compute the AUC
                    auc = self.auc_eval(probs, labels)
                    # Compute the accuracy
                    acc = self.acc_eval(probs, labels)
                    # Compute total auc and accuracy in the whole batch and add to train auc and train_acc
                    val_auc += auc.item() * inputs.size(0)
                    val_acc += acc.item() * inputs.size(0)
                    print("Val Batch: {:03d}, Val Loss: {:.4f}, Batch Val AUC {:.4f}, Acc: {:.4f}".format(j, loss.item(), auc.item(), acc.item()))    
    
            # Find average training loss and training accuracy
            avg_train_loss = train_loss/self.train_data_size 
            avg_train_acc = train_acc/self.train_data_size
            avg_train_auc = train_auc/self.train_data_size

            self.train_loss_values.append(avg_train_loss)
            self.train_acc_values.append(avg_train_acc)
            self.train_auc_values.append(avg_train_auc)

            # Find average validation loss and validation accuracy
            avg_val_loss = val_loss/self.val_data_size 
            avg_val_acc = val_acc/self.val_data_size
            avg_val_auc = val_auc/self.val_data_size

            self.val_loss_values.append(avg_val_loss)
            self.val_acc_values.append(avg_val_acc)
            self.val_auc_values.append(avg_val_auc)

            epoch_end = time.time()
            print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%\nValidation : Loss : {:.4f}, Accuracy: {:.4f}%\nTime: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_val_loss, avg_val_acc*100, epoch_end-epoch_start))

            if epoch == 0:
                self.best_loss = self.val_loss_values[-1]
                self.best_model_auc = self.val_auc_values[-1]
                self.best_model_acc = self.val_acc_values[-1]

            if round(self.val_loss_values[-1], 4) < round(self.best_loss, 4):
                self.best_loss = self.val_loss_values[-1]
                self.best_model_auc = self.val_auc_values[-1]
                self.best_model_acc = self.val_acc_values[-1]
                last_improvement = epoch
                self.save(model_save_path)

            elif epoch - last_improvement > round(0.2*self.num_epochs,0):
                print(f'Validation loss has not improved for {round(0.2*self.num_epochs,0)} epochs, stopping training.\n')
                break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.val_loss_values[-1] < self.best_loss:
            print('Last Model Update is not the one with smallest validation loss.')
            print('Loading lowest validation loss model.')
            self.load(model_save_path)


    def test(self, load_from_path=False, model_path=None):
        # Checks if user wants to load a model from a file to test
        if load_from_path and model_path is not None:
            self.load(model_path)
        elif load_from_path:
            raise TypeError("The load_from_model parameter was set to True, but no model path was given.")
        
        # Size of Data, to be used for calculating Average Loss and Accuracy
        self.test_data_size = len(self.datasets['test'])

        # Use DataLoader with batch size equal to the full dataset for a single full size test batch 
        self.test_loader = DataLoader(self.datasets['test'], batch_size=self.test_data_size)

        # Test - No gradient tracking needed
        with torch.no_grad():
            # Set to evaluation mode
            self.model.eval()
            # Test loop
            for inputs, labels in self.test_loader:
                # Send inputs and labels to GPU
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass - compute outputs on test data using the model
                self.test_outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(self.test_outputs, labels)
                # Compute the test loss
                self.test_loss = loss.item()
                # Applies Softmax to obtain the probabilities
                self.test_probs = nn.Softmax(dim=1)(self.test_outputs)[:,1]
                # Compute the AUC
                self.test_auc = self.auc_eval(self.test_probs, labels).item()

                print("Test: Loss: {:.4f}, Test AUC: {:.4f}".format(self.test_loss, self.test_auc))

    def show_test_pred(self, test_image_path=None):
        # Reads a user specified image path for the model to predict
        if test_image_path is not None:
            test_img = Image.open(test_image_path).convert('RGB')
            test_img_tensor = self.datasets['test'].data_transforms['test'](test_img).unsqueeze(0).to(device)
        
        # If no path is specified, get a random image from the test set
        else:
            rnd_idx = np.random.randint(0, self.datasets['test'].__len__()+1)
            test_img = Image.open(self.datasets['test'].image_paths[rnd_idx]).convert('RGB')
            test_img_tensor = self.datasets['test'].data_transforms['test'](test_img).unsqueeze(0).to(device)
            test_img_label = self.datasets['test'].label_values[rnd_idx]

        with torch.no_grad():
            self.model.eval()
            # Model output
            out = self.model(test_img_tensor)
            _, test_pred = torch.max(out.data, 1)
            probs = nn.Softmax(dim=1)(out)

        results = []
        for i, lab in enumerate(self.datasets['test'].classes):
            line = f"{lab}: {round(probs[0][i].item(), 3)}"
            results.append(line)

        if test_image_path is None:
            results.append(f"\nTrue Label: {self.datasets['test'].classes[test_img_label]}")

        title = '\n'.join(results)

        plt.figure(figsize=(6,6))

        plt.imshow(test_img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.show()     

def cross_val_loop(train_val_df, test_df, tuning_params, scoring='auc', k=5, model_save_path='best_cv_model.pth'):
    """
        Implements a cross validation loop for the image classifier.

        User can specify the scoring metric for the choosing of the
        best model between the cross entropy loss and the AUC.

        The parameter k is the number of folds to be implemented.

        The tuning_params argument requires a dictionary with the 
        keys as parameters and values as the respective parameter
        value to be used. The following tuning parameters are 
        required:
            - optimizer (name from torch.optim)
            - lr        (learning rate for the optimizer)
            - L2        (weight decay, L2 penalizer)
            - batch_size
            - num_epochs
    """
    kf = KFold(n_splits=k)

    cv_val_loss_values = []
    cv_val_auc_values = []
    
    best_score = None

    print('Running Trial with:')
    for key, value in tuning_params.items():
        print(f"{key}: {value}")

    for fold, (train_index, val_index) in enumerate(kf.split(train_val_df)):
        print("\n\nStarting fold {}/{}\n".format(fold+1, k))
        # Creates the train and val datasets for the fold
        train_fold = train_val_df.iloc[train_index]
        val_fold = train_val_df.iloc[val_index]

        model = ImageClassifier(
            train_df=train_fold,
            val_df=val_fold,
            test_df=test_df,
            params={
                'optimizer':tuning_params['optimizer'],
                'lr':tuning_params['lr'],
                'L2':tuning_params['L2']
            }
        )

        model.train(num_epochs=tuning_params['num_epochs'], batch_size=tuning_params['batch_size'])

        cv_val_loss_values.append(model.best_loss)
        cv_val_auc_values.append(model.best_model_auc)

        if best_score is None:
            best_model = model
            if scoring =='auc':
                best_score = cv_val_auc_values[-1]
            elif scoring=='loss':
                best_score = cv_val_loss_values[-1]

        elif scoring =='auc':
            if round(cv_val_auc_values[-1], 4) < round(best_score, 4):
                best_score = cv_val_auc_values[-1]
                best_model = model
            
        elif scoring=='loss':
            if round(cv_val_loss_values[-1], 4) < round(best_score, 4):
                best_score = cv_val_loss_values[-1]
                best_model = model
    

    best_model.save(model_save_path)
    mean_cv_val_loss = np.array(cv_val_loss_values).mean()
    mean_cv_val_auc = np.array(cv_val_auc_values).mean()

    print("Finished {k}-fold CV")
    print(f"Mean CV Loss: {mean_cv_val_loss}")
    print(f"Mean CV AUC: {mean_cv_val_auc}")

    return best_model, mean_cv_val_loss, mean_cv_val_auc

