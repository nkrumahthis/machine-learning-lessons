import torch
import matplotlib.pyplot as plt

# Generate synthetic data
torch.manual_seed(42)  # For reproducibility
X = torch.linspace(0, 10, 100).unsqueeze(1)  # Shape (100, 1)
true_w, true_b = 2.0, 5.0  # y = 2x + 5
y = true_w * X + true_b + torch.randn_like(X) * 2  # Adding noise

# Initialize parameters
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Hyperparameters
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = w * X + b  # Linear model

    # Compute loss (MSE)
    loss = ((y_pred - y) ** 2).mean()

    # Backpropagation (compute gradients)
    loss.backward()

    # Update parameters (Gradient Descent)
    with torch.no_grad():  # Stop tracking gradients
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Zero gradients after updating
        w.grad.zero_()
        b.grad.zero_()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, w: {w.item():.4f}, b: {b.item():.4f}")

# Final parameters
print(f"Trained parameters: w = {w.item():.4f}, b = {b.item():.4f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(X.numpy(), y.numpy(), label="Data points")  # plot the data points
plt.show()