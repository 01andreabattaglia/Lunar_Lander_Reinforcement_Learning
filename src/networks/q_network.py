import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A Q-Network implemented with PyTorch.
    
    Architecture:
    - Input: State with shape (8,)
    - Hidden1: 128 neurons + ReLU activation
    - Hidden2: 256 neurons + ReLU activation
    - Hidden3: 512 neurons + ReLU activation
    - Output: 4 Q-values with Linear activation (no activation)
    
    Contract:
    - Input: State with shape (8,) containing real-valued numbers
    - Output: Q-values for each of 4 actions with shape (4,)
             Q(s,0), Q(s,1), Q(s,2), Q(s,3)
    - Q-values are NOT probabilities, can be positive or negative
    """
    
    def __init__(self, input_size=8, output_size=4):
        """
        Initialize the Q-Network.
        
        Args:
            input_size (int): Size of input state (default: 8 for LunarLander)
            output_size (int): Number of actions (default: 4 for LunarLander)
        """
        super(QNetwork, self).__init__()
        
        # Detect device (prioritize CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(256, 512)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(512, output_size)
        # No activation for output layer (Linear)
        
        # Move network to device
        self.to(self.device)
    
    def forward(self, state):
        """
        Compute Q-values for a given state.
        
        Args:
            state (torch.Tensor or np.ndarray): Input state with shape (8,) or (batch_size, 8)
        
        Returns:
            torch.Tensor: Q-values for each action with shape (4,) or (batch_size, 4)
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
        q_values = self.fc4(x)
        
        # Return squeezed tensor if input was 1D
        if q_values.shape[0] == 1:
            return q_values.squeeze(0)
        
        return q_values
    
    def predict(self, state):
        """
        Predict Q-values for a given state (alias for forward).
        
        Args:
            state (torch.Tensor or np.ndarray): Input state with shape (8,)
        
        Returns:
            torch.Tensor: Q-values for each action with shape (4,)
        """
        with torch.no_grad():
            return self.forward(state)