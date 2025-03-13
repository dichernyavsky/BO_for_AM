import numpy as np
import torch

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.range_min, self.range_max = feature_range

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()  
        self.min = torch.min(X, dim=0)[0]
        self.max = torch.max(X, dim=0)[0]

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()  
        # map to the range [a, b]
        return (X - self.min) * (self.range_max - self.range_min) / (self.max - self.min) + self.range_min

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()  # change to double precision
        # inverse map from [a, b] to the oroginal range
        return (X - self.range_min) * (self.max - self.min) / (self.range_max - self.range_min) + self.min

    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double() 
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double() 
        return (X - self.mean) / (self.std + 1e-10)  # adding small constant to avoid division by zero

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double() 
        return X * self.std + self.mean
    


class GPDataHandler:
    """
    This class handles the data preprocessing for Gaussian Process models.
    """
    def __init__(self, df, coordinates, main_target_name, std_density, bounds_density_scaling=[0.1, 1], bounds_bur_model=[0.1, 1.1]):
        self.df = df
        self.coordinates = coordinates
        self.target = main_target_name
        self.std_density = std_density
        self.bounds_density_scaling = bounds_density_scaling
        self.bounds_bur_model = bounds_bur_model

        # Scalers for data normalization
        self.scaler_X = MinMaxScaler(feature_range=(bounds_density_scaling[0], bounds_density_scaling[1]))
        self.scaler_Y = StandardScaler()

        # Setup data for density and bur models
        self.setup_data_density()
        self.setup_data_bur()

    def setup_data_density(self):
        """
        Prepares and scales the density data from the dataframe.
        """
        X = self.df[self.coordinates].values
        Y = self.df[self.target].values.reshape(-1, 1)
        
        # Convert to torch tensors
        X = torch.tensor(X).double()
        Y = torch.tensor(Y.flatten()).unsqueeze(-1).double()
        
        # Save original data
        self.X_original_density = X
        self.Y_original_density = Y

        # Scale data
        self.X_scaled_density = torch.tensor(self.scaler_X.fit_transform(X.numpy())).double()
        self.Y_scaled_density = torch.tensor(self.scaler_Y.fit_transform(Y.numpy())).double()

    def setup_data_bur(self, n_step=15):
        """
        Prepares data for the BUR model.
        """
        X_bur, Y_bur = self.give_X_bur(n_step)
        self.X_bur = X_bur
        self.Y_bur = Y_bur

    @staticmethod
    def give_grid(bounds, n_step):
        """
        Generates a grid for the given bounds and number of steps.
        """
        x = torch.linspace(bounds[0], bounds[1], n_step)
        xx, yy, zz = torch.meshgrid([x, x, x], indexing='ij')
        
        X_grid = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], -1)
        
        return X_grid

    def give_X_bur(self, n_step=15):
        """
        Generates the X and Y matrices for the BUR model based on the defined grid.
        """
        X_bur = GPDataHandler.give_grid(self.bounds_bur_model, n_step)
        Y_bur = X_bur[:, 1] * X_bur[:, 2]  # build-up rate = hatch distance * laser velocity
        return X_bur.double(), Y_bur.double().unsqueeze(-1)