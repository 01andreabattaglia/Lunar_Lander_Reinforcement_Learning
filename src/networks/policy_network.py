import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    A Policy Network for REINFORCE algorithm implemented with PyTorch.
    
    Architecture:
    - Input: State with shape (8,)
    - Hidden1: 16 neurons + ReLU activation
    - Hidden2: 16 neurons + ReLU activation
    - Hidden3: 16 neurons + ReLU activation
    - Output: 4 action probabilities with Softmax activation
    
    Contract:
    - Input: State with shape (8,) containing real-valued numbers
    - Output: Probability distribution over 4 actions with shape (4,)
             π(a=0|s), π(a=1|s), π(a=2|s), π(a=3|s)
    - Probabilities sum to 1.0
    """
    
    def __init__(self, input_size=8, output_size=4):
        """
        Initialize the Policy Network.
        
        Args:
            input_size (int): Size of input state (default: 8 for LunarLander)
            output_size (int): Number of actions (default: 4 for LunarLander)
        """
        super(PolicyNetwork, self).__init__()
        
        # Detect device (prioritize CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fc1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(16, 16)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(16, output_size)
        # Softmax applied in forward pass
        
        # Move network to device
        self.to(self.device)
    
    def forward(self, state):
        """
        Compute action probabilities for a given state.
        
        Args:
            state (torch.Tensor or np.ndarray): Input state with shape (8,) or (batch_size, 8)
        
        Returns:
            torch.Tensor: Action probabilities with shape (4,) or (batch_size, 4)
        """
        # Convert numpy array to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Move state to the same device as the network
        state = state.to(self.device)
        
        # Ensure 2D shape for batch processing
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        logits = self.fc4(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        return probs
    
    def predict(self, state):
        """
        Predict action probabilities for a given state (alias for forward).
        
        Args:
            state (torch.Tensor or np.ndarray): Input state with shape (8,)
        
        Returns:
            torch.Tensor: Action probabilities with shape (4,)
        """
        with torch.no_grad():
            return self.forward(state)
