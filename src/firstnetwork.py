inputs: list[int] = [1,2,3,4]
targets: list[int] = [2,4,6,8]

# Weight, Slope, Gradient all three are the same thing!!!

# randomly initialized
w: float = 0.1
learning_rate: float = 0.1

# Neural Network
def predict(i:int):
    return w * i

# Train the Neural Network
for _ in range(25):
    # Prediction:
    predictions: list[float] = [predict(i) for i in inputs]
    # Error:
    errors: list[float] = [ target - prediction for prediction, target in zip(predictions, targets)]
    # Cost Function: 
    cost: list[float] = sum(errors) / len(targets)
    print(f"weight: {w:.2f} | cost: {cost:.2f}")
    print(">> Backpropagation: updates weight")
    # Backpropagation
    w += learning_rate * cost

# Batch Gradient Descent: where each training sample is processed before updating the weight

# Test the Neural Network
test_inputs = [5,6,7]
test_targets = [10,12,14]

predictions: list[float] = [predict(i) for i in test_inputs]
errors = [target - prediction for prediction, target in zip(predictions, test_targets)]
cost = sum(errors) / len(test_targets)
print(cost)

for inp, target, pred in zip(test_inputs, test_targets, predictions):
    print(f"inp:{inp:.2f} | target:{target:.2f} | pred:{pred:.2f}")