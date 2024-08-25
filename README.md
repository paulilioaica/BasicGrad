# BasicGrad


<p align="center">
  <img src="./media/image.webp" width="300px" height="300px"/>
</p>

# BasicGrad: A Minimalist Autograd Engine

Welcome to **BasicGrad**, a minimalist autograd engine inspired by the famous [micrograd](https://github.com/karpathy/micrograd) project by Andrej Karpathy. This library implements a dynamic computational graph with automatic differentiation in just a few lines of Python code. It's designed for educational purposes, making it easy to understand the core concepts behind deep learning frameworks like PyTorch.


## 🧠 Overview of the Logic

BasicGrad is designed around the concept of chaining variables through operations, building a dynamic computational graph that can be used to compute gradients for any scalar output. Here's a breakdown of how it all works:

### 1. **Chaining Variables with Operations**
   - **Core Idea**: In BasicGrad, every operation (like addition or multiplication) creates a new `Variable` object. These operations chain together to form a computational graph, where each node (a `Variable`) knows about its predecessors. These operations are attached their specific backward functions.
   - **Why It Matters**: This chaining allows us to keep track of the relationships between variables, which is essential for backpropagation and gradient computation.

### 2. **Custom Derivatives for Each Operation**
   - **Manual Gradient Calculation**: Each operation in BasicGrad has a specific derivative (or gradient) that needs to be computed manually. For instance, in addition, the gradient is simply passed along to both operands, while in multiplication, the gradient is multiplied by the value of the other operand.
   - **Flexible and Extensible**: This design makes BasicGrad highly flexible, allowing you to easily extend it with new operations by defining their forward pass and corresponding backward (gradient) logic.

### 3. **Backpropagation: Traversing the Graph**
   - **Topological Order**: To compute the gradients, BasicGrad first constructs a topological order of nodes in the graph, ensuring that each node is processed after its dependencies.
   - **Reverse Pass**: Backpropagation then works by traversing this order in reverse, applying the chain rule to compute the gradient of each node (variable) with respect to the final output.

### 4. **Gradient Flow and Accumulation**
   - **Gradient Accumulation**: During backpropagation, gradients are accumulated at each node. For example, if a variable contributes to multiple downstream operations, its gradient is the sum of its contributions to those operations.
   - **Automatic Differentiation**: By leveraging the computational graph, BasicGrad automates the differentiation process, allowing you to easily compute gradients for any scalar output.

### 5. **Building Neural Networks**
   - **Layer by Layer**: With these core principles, you can start building neural network layers, chaining them together using the same operations and backpropagation logic.
   - **Activation Functions**: Implementing activation functions as operations allows them to seamlessly integrate into the graph, contributing to the gradient flow just like any other operation.

### Why This Matters

Understanding this logic is key to appreciating the power and flexibility of autograd engines like BasicGrad. By manually defining derivatives for each operation, you're not only learning the mathematics behind deep learning but also gaining insight into how more complex frameworks like PyTorch and TensorFlow operate under the hood.


## 🌟 Features

- **Dynamic Computational Graph**: Build and manage computational graphs on-the-fly with automatic differentiation.
- **Modular Operations**: Easily extendable with new operations. The current implementation supports addition (`+`) and multiplication (`*`), but you can add more by following the simple `Operation` interface.
- **Operator Overloading**: Perform operations using Python's native arithmetic operators, making the code intuitive and easy to use.
- **Backpropagation**: Efficiently compute gradients for scalar outputs using the backpropagation algorithm.
- **Pythonic and Elegant**: Written in a clean and modular style, leveraging Python’s object-oriented and functional capabilities.

## 🚀 Getting Started

### Installation

You can clone this repository and use it directly. No dependencies are required!

```bash
git clone https://github.com/paulilioaica/BasicGrad
cd BasicGrad
```

### Usage Example

Here's a simple example to get you started:

```python
from diff import Variable

# Create variables
a = Variable(2)
b = Variable(3)

# Perform operations
c = a + b
d = a * b

# Perform backpropagation
c.backward()
d.backward()

# Inspect gradients
print(a)  # Variable(data=2, grad=4)
print(b)  # Variable(data=3, grad=3)
print(c)  # Variable(data=5, grad=1)
print(d)  # Variable(data=6, grad=1)
```

### How It Works

#### Variable Class

The `Variable` class represents a node in the computational graph. Each `Variable` holds a value (`data`), a gradient (`grad`), and references to its predecessors in the graph (`_prev`). The class handles:

- **Operator Overloading**: Through methods like `__add__` and `__mul__`, which enable Pythonic arithmetic operations.
- **Backpropagation**: Using the `backward()` method, which computes the gradient for each `Variable` by traversing the computational graph in reverse order.

#### Operation Class

The `Operation` class is an abstract base class that defines how each operation should be applied and how the backward pass should be handled. Two concrete implementations are provided:

- **AddOperation**: Handles addition and the corresponding gradient calculations.
- **MulOperation**: Handles multiplication and the corresponding gradient calculations.
--- 
### Extending BasicGrad

Want to add more operations? Simply extend the `Operation` class and define the `forward` and `_build_backward_function` methods. Here's a quick example of how you might add a subtraction operation:

```python
class SubOperation(Operation):
    def forward(self, a, b):
        return Variable(a.data - b.data, (a, b))

    def _build_backward_function(self, a, b, out):
        def _backward():
            a.grad += out.grad
            b.grad -= out.grad
        return _backward
```

Then, integrate it into the `Variable` class:

```python
class Variable:
    # Existing methods...

    def __sub__(self, other):
        return self._apply(SubOperation(), other)
```
---

### Why BasicGrad?

BasicGrad is a simplified and educational framework that offers a glimpse into how deep learning libraries work under the hood. It's a great tool for:

- **Learning**: Understand the basics of automatic differentiation and computational graphs.
- **Prototyping**: Quickly experiment with new ideas and custom operations.
- **Teaching**: Use it as a teaching aid to explain concepts in a clear and concise manner.


## 🔮 Features to Come

BasicGrad is just the beginning! Here are some exciting features planned for future development:

- **Complete Set of Operations**: Implement a comprehensive range of mathematical operations, including subtraction, division, exponentiation, and more.
  
- **Activation Functions**: Add support for common activation functions such as ReLU, Sigmoid, Tanh, and more, to enable the construction of neural networks.
  
- **Neural Network Architectures**: Build foundational tools for constructing various neural network architectures, including fully connected layers, convolutional layers, and others.
  
- **Graph Visualization**: Develop tools to visualize computational graphs, making it easier to understand the structure of complex models and track gradients during backpropagation.

--- 

## 📚 Inspired By

This project draws inspiration from Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) project, which implements a tiny autograd engine in just 100 lines of code. BasicGrad takes the concepts from micrograd and adds additional modularity and extensibility.

## 🤝 Contributing

Contributions are welcome! Whether it's adding new operations, improving documentation, or optimizing the code, your help is appreciated. Feel free to open issues or submit pull requests.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
