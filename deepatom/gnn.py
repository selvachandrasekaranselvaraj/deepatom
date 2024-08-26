import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class GNNModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = torch_geometric.nn.global_mean_pool(x, batch)  # Pooling layer
        x = self.fc(x)
        return x

def structure_to_graph(atoms):
    num_atoms = len(atoms)
    num_node_features = 1  # Example number of node features

    # Node features (example: atomic numbers)
    x = torch.tensor([atom.number for atom in atoms], dtype=torch.float).view(-1, 1)
    
    # Edge indices (example: fully connected graph)
    edge_index = torch.combinations(torch.arange(num_atoms), r=2).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    return data
