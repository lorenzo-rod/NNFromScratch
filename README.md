# Neural network from scractch
This is a personal project in which I created a neural network using just numpy (no tensorflow, 
no pytorch, no keras). It was done to test my knowledge on how neural networks work and potentially
help people in understanding them.

I only used matplotlib for visualization and skelearn to access the mnist dataset, but no ML model from
sklearn is used.

## Table of Contents
- [Neural network from scractch](#neural-network-from-scractch)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [4. **Usage**](#4-usage)
  - [Usage](#usage)

## Installation
Clone the repository:
```bash
git clone https://github.com/lorenzo-rod/NNFromScratch.git
pip install -r requirements.txt


### 4. **Usage**

```markdown
## Usage
The project contains three files in which you can see how the network is trained for different problems:
- xor_training.py: Trains a network to learn a xor gate (used for model validation).
- sine_training.py: Trains a network to fit a sine wave.
- mnist_training.py: Trains a network for the mnist dataset.

Additionally, there is a filed called mnist visualization in which you can observe how a trained model behaves
in a test set. The model accuracy in this test set is 90.01%.