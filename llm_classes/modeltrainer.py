# Import necessary libraries and modules
import torch as t

# Training proces
class ModelTrainer:
    # Constructor to initialize the ModelTrainer object with necessary components
    def __init__(
        self, model, train_data, test_data, optimizer, loss_function, device, cfg
    ):
        self.cfg = cfg  # Configuration parameters
        self.model = model  
        self.train_data = train_data  
        self.test_data = test_data  
        self.optimizer = optimizer  # Optimizer for updating model weights
        self.loss_function = loss_function 
        self.device = device  # Device (CPU or GPU) on which to perform computations

        self.model.train()  # Set the model to training mode

    # Method to perform a single epoch of training
    def train_epoch(self):
        losses = 0  # Initialize total loss for the epoch
        for (inputs, targets) in self.train_data:  # Iterate over batches of training data
            inputs = inputs.to(self.device)  # Move inputs to the specified device
            targets = targets.to(self.device)  
            output = self.model(inputs)  # Forward pass: compute model's output
            loss = self.loss_function(output, targets)  # Compute loss between output and targets
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backward pass: compute gradients
            self.optimizer.step()  # Update model weights
            losses += loss.item()  # Accumulate the loss
        return losses / len(self.train_data)  # Return average loss per batch

    # Method to evaluate the model's performance on test data
    def evaluate(self):
        self.model.eval()  # Set the model to evaluation mode
        passed = 0  # Initialize counter for correct predictions
        with t.no_grad():  # Disable gradient computation
            for (inputs, targets) in self.test_data:  # Iterate over batches of test data
                inputs = inputs.to(self.device)  
                targets = targets.to(self.device) 
                outputs = self.model(inputs) 
                # Check if the predicted class (max logit) matches the target class
                if outputs.argmax() == targets.argmax():
                    passed += 1  # Increment correct prediction count
        # Calculate accuracy as the ratio of correct predictions to total samples
        accuracy = passed / len(self.test_data.dataset)
        return accuracy  

    # Method to conduct the entire training process over multiple epochs
    def train(self, epochs):
        for epoch in range(epochs):  
            train_loss = self.train_epoch()  # Train for one epoch and get training loss
            accuracy = self.evaluate()  # Evaluate the model to get accuracy
            # Print epoch, training loss, and test accuracy
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Accuracy: {accuracy:.4f}"
            )