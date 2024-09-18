DeepAtom is a Python package designed for calculating and utilizing various atomic descriptors for machine learning applications in materials science. It includes functionality for:

- **Descriptors Calculation**: Computes a range of atomic descriptors such as Coulomb matrices, sine matrices, Ewald sum matrices, Atom-centered Symmetry Functions (ACSF), Smooth Overlap of Atomic Positions (SOAP), and Many-body Tensor Representation (MBTR).
- **Graph Neural Networks (GNNs)**: Implements a GNN model for predicting properties based on atomic structures.
- **Utilities**: Provides functions to handle LAMMPS output trajectory files and convert them into formats suitable for descriptor calculations.
- **Training and Prediction**: Allows training of GNN models and prediction of material properties with saved models.

## Installation

To install DeepAtom, you can use pip:

```bash
pip install deepatom
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/yourusername/deepatom.git
cd deepatom
pip install -e .
```

## Usage Examples

Here are some examples of how to use DeepAtom:

### Basic Usage

```python
from deepatom.main import DeepAtom

# Create a DeepAtom instance
deepatom = DeepAtom()
```

### Loading Data

DeepAtom can read LAMMPS trajectory files:

```python
# Load data from a LAMMPS trajectory file
deepatom.load_data()
```

Note: Make sure to provide the LAMMPS trajectory file as a command-line argument when running your script.

### Calculating Descriptors

You can calculate various descriptors:

```python
# Calculate ACSF and SOAP descriptors
deepatom.calculate_descriptors()

# Print the shapes of the calculated descriptors
print("SOAP descriptors shape:", np.array(deepatom.soap_descriptors).shape)
print("ACSF descriptors shape:", np.array(deepatom.acsf_descriptors).shape)
```

### Preparing Data for Training

To prepare your data for training:

```python
# Prepare data for training
deepatom.prepare_data_for_training()
```

This method combines different descriptors, splits the data into training and testing sets, and scales the features.

### Training a Model

To train a Graph Neural Network model:

```python
# Train the model
deepatom.train(epochs=10, learning_rate=0.001)
```

This will train the model and save it as 'deepatom_model.pth'.

### Making Predictions

After training or loading a model, you can make predictions:

```python
# Load a trained model
deepatom.load_model('deepatom_model.pth')

# Make predictions
deepatom.predict()
```

This will output the accuracy of the model on the test set.

## Full Example

Here's a complete example that demonstrates the entire workflow:

```python
from deepatom.main import DeepAtom

def main():
    deepatom = DeepAtom()
    deepatom.load_data()
    deepatom.calculate_descriptors()
    deepatom.prepare_data_for_training()
    deepatom.train(epochs=10, learning_rate=0.001)
    deepatom.predict()

if __name__ == "__main__":
    main()
```

Run this script with a LAMMPS trajectory file as a command-line argument:

```bash
python your_script.py your_trajectory_file.lmp
```

For more detailed information on each function and its parameters, please refer to the source code and documentation.