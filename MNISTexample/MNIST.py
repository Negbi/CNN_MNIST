import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Define hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 1

# Step 2: Prepare the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1] for faster computing & smoother convergence
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

print("dataset train number of images", len(train_dataset))
print("dataset test number of images", len(test_dataset))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Step 3: Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # first convolution layer, which contains 1 input (images are grayscale) and 16 feature maps by the kernel, which is 3x3 with 1 layer of padding around the picture
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # second conv layer
        self.pool = nn.MaxPool2d(2, 2)  # downsample by 2 using max pooling which will keep the main extraction imported data
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # fully connected layer
        self.fc2 = nn.Linear(128, 10)  # output layer

    def forward(self, x): # the actual workflow/architecture of the forward step of the layer
        x = self.pool(torch.relu(self.conv1(x))) # take x, conv it using conv1 (on self layer which is the NN we built up here), RelU each output (sum it up and
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 4: instantiate the model, define loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #use GPU instead of CPU for faster computing
model = SimpleCNN().to(device) #match our model to the simpleCNN we just created
criterion = nn.CrossEntropyLoss() #usually CEL doesnt saturate as much in SGD as MSE
optimizer = optim.SGD(model.parameters(), lr=learning_rate) #optimze method, use SGD with our parameters models and the defined lr

# Step 5: Train the model
model.train() #tells the model that you are training it. This helps inform layers such as Dropout and BatchNorm to be activated for training (we dont want them on during testing)
running_loss = 0.0 #loss value
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device) #insert the corrisponding image and label from train_loader

    # reset gradient
    optimizer.zero_grad()

    # Forward pass
    outputs = model(images) # all the outputs that we got
    loss = criterion(outputs, labels) #calculate cross entropy loss between outputs and labels


    loss.backward() # return the loss gradient calculated during backprop, specifically CNN backprop
    optimizer.step() #update model's parameters based on the gradient loss in the backprop

    running_loss += loss.item() #update the loss of current run

print(f"Epoch [{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 6: Test the model
model.eval() #this time eval = test mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {100 * correct / total:.2f}%")

# Step 7: Save the model
torch.save(model.state_dict(), "simple_cnn_mnist_sgd.pth")
print("Model saved to simple_cnn_mnist_sgd.pth")