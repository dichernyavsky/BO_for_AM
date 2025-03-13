import torch
import gpytorch
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll



class GPModelManager:
    """
    This class manages the Gaussian Process models for density and BUR prediction.
    """
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.density_model = None
        self.bur_model = None

    def train_density_model(self):
        """
        Trains a Gaussian Process model for density prediction.
        """
        # Define model with fixed noise based on standard deviation
        train_Yvar = torch.full_like(self.data_handler.Y_scaled_density, 
                                     float((self.data_handler.std_density**2) / (self.data_handler.scaler_Y.std ** 2))) 

        # Setup the Gaussian Process model
        self.density_model = FixedNoiseGP(train_X=self.data_handler.X_scaled_density, 
                                     train_Y=self.data_handler.Y_scaled_density, 
                                     train_Yvar=train_Yvar,
                                     mean_module=gpytorch.means.ZeroMean(),
                                     covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,
                                                                                                              ard_num_dims=self.data_handler.X_scaled_density.shape[1])))

        # Train model 
        mll = ExactMarginalLogLikelihood(self.density_model.likelihood, self.density_model)
        fit_gpytorch_mll(mll)

    def train_bur_model(self):
        """
        Trains a Gaussian Process model for BUR prediction.
        """
        # Fixed noise variance
        train_Yvar = torch.full_like(self.data_handler.Y_bur, 0.0001)

        # Setup the Gaussian Process model
        self.bur_model = FixedNoiseGP(train_X=self.data_handler.X_bur, 
                                 train_Y=self.data_handler.Y_bur, 
                                 train_Yvar= train_Yvar)

        # Train model
        mll = ExactMarginalLogLikelihood(self.bur_model.likelihood, self.bur_model)
        fit_gpytorch_mll(mll)

    def predict_density(self, X, return_in_original_scale=True, return_std=False):
        """
        Predicts density using the trained model.
        """
        self.density_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predicted_mean = self.density_model(X).mean
            predicted_var = self.density_model(X).variance
            predicted_std = torch.sqrt(predicted_var)

        if return_in_original_scale:
            predicted_mean = self.data_handler.scaler_Y.inverse_transform(predicted_mean.numpy().reshape(-1, 1)).reshape(-1)
            predicted_std = predicted_std * self.data_handler.scaler_Y.std

        if return_std:
            return predicted_mean, predicted_std
        else:
            return predicted_mean

    def predict_bur(self, X, return_std=False):
        """
        Predicts BUR using the trained model.
        """
        self.bur_model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predicted_mean = self.bur_model(X).mean
            predicted_var = self.bur_model(X).variance
            predicted_std = torch.sqrt(predicted_var)

        if return_std:
            return predicted_mean, predicted_std
        else:
            return predicted_mean
        
    def setup_model(self):
        print('Training density model...')
        self.train_density_model()
        
        print('Training BUR model...')
        self.train_bur_model()

        
        return ModelListGP(self.density_model, self.bur_model)