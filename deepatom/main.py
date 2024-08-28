import torch
import torch.nn.functional as F
from deepatom.utils import read_lammps_trajectory
from deepatom.descriptors import (
    calculate_coulomb_matrix,
    sine_matrix,
    calculate_ewald_sum,
    calculate_acsf,
    calculate_soap_descriptors,
    calculate_mbtr_descriptors
)
from deepatom.gnn import structure_to_graph, GNNModel
import numpy as np
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DeepAtom:
    def __init__(self):
        self.atoms = None
        self.trajectory_file = None
        self.atomic_numbers = None
        self.lattice_vectors = None
        self.positions = None
        self.atoms = None
        self.structure = None
        self.model = None
        self.scaler = None

    def load_data(self):
        self.atoms = read_lammps_trajectory()
        self.atomic_numbers = np.array([atom.numbers for atom in self.atoms])
        self.lattice_vectors = np.array([atom.cell for atom in self.atoms] )     
        self.positions = np.array([atom.positions for atom in self.atoms])
        self.structure = Structure(self.lattice_vectors, self.atomic_numbers, self.positions)

    def calculate_descriptors(self):
        #self.cm_matrix = calculate_coulomb_matrix(self.structure)
        #self.sine_mat = sine_matrix(self.positions)
        #self.ewald_matrix = calculate_ewald_sum(self.structure)
        self.acsf_descriptors = calculate_acsf(self.atoms)
        self.soap_descriptors = calculate_soap_descriptors(self.atoms)
        self.mbtr_features = calculate_mbtr_descriptors(self.positions, self.atomic_numbers)

    def prepare_data_for_training(self):
        self.descriptors = np.concatenate([
            self.cm_matrix.flatten(),
            self.sine_mat.flatten(),
            self.ewald_matrix.flatten(),
            self.acsf_descriptors.flatten(),
            self.soap_descriptors.flatten(),
            self.mbtr_features.flatten()
        ])
        # Example labels, replace with actual labels if available
        self.labels = np.random.randint(0, 2, size=(self.descriptors.shape[0],))  # Binary classification

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(self.descriptors, self.labels, test_size=0.2, random_state=42)

        # Scale features
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        self.train_data = torch.tensor(X_train, dtype=torch.float32)
        self.train_labels = torch.tensor(y_train, dtype=torch.long)
        self.test_data = torch.tensor(X_test, dtype=torch.float32)
        self.test_labels = torch.tensor(y_test, dtype=torch.long)

    def train(self, epochs=10, learning_rate=0.001):
        self.model = GNNModel(num_node_features=self.train_data.shape[1], num_classes=2)  # Adjust num_classes as needed
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.train_data)
            loss = F.cross_entropy(output, self.train_labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

        # Save the trained model
        torch.save(self.model.state_dict(), 'deepatom_model.pth')
        print("Model saved as 'deepatom_model.pth'")

    def load_model(self, model_path='deepatom_model.pth'):
        self.model = GNNModel(num_node_features=self.train_data.shape[1], num_classes=2)  # Adjust num_classes as needed
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded from '{model_path}'")

    def predict(self):
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Please train or load a model first.")
        with torch.no_grad():
            predictions = self.model(self.test_data)
            _, predicted_classes = torch.max(predictions, 1)
            accuracy = (predicted_classes == self.test_labels).float().mean()
            print(f'Accuracy: {accuracy.item() * 100:.2f}%')

    def print_results(self):
        print("Coulomb Matrix:\n", self.cm_matrix)
        print("Sine Matrix:\n", self.sine_mat)
        print("Ewald Sum Matrix:\n", self.ewald_matrix)
        print("ACSF Descriptors:\n", self.acsf_descriptors)
        print("SOAP Descriptors:\n", self.soap_descriptors)
        print("MBTR Descriptors:\n", self.mbtr_features)

def main():
    # Define your parameters
    trajectory_file = 'your_trajectory_file.lmp'  # Replace with your LAMMPS trajectory file path
    atomic_numbers = np.array([13, 8, 13])  # Example atomic numbers
    lattice_vectors = np.eye(3)  # Replace with actual lattice vectors

    deepatom = DeepAtom(trajectory_file, atomic_numbers, lattice_vectors)
    deepatom.load_data()
    deepatom.calculate_descriptors()
    deepatom.prepare_data_for_training()
    deepatom.train(epochs=10, learning_rate=0.001)
    deepatom.predict()
    deepatom.print_results()

if __name__ == "__main__":
    main()

