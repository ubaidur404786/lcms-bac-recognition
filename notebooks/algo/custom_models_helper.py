import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import copy  # for safely saving best model weights

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch



''' OPTIMIZED PARAMETERS '''

LATENT_DIM = 64
NUM_HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 256
DROPOUT_RATE = 0.5 
BATCH_SIZE = 64
EPOCHS = 2000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NOISE_FACTOR = 0.15



'''Dataset class'''
class LCMSDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]

''' Early Stopping Class '''
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



'''Autoencoder model'''
class AutoEncoder(nn.Module):
    def __init__(self, input_features, latent_features=LATENT_DIM):
        super().__init__()
        
        # Encoder: Compresses high-dim input -> 64 latent features
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),  # Added BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, latent_features) # Bottleneck (z)
        )

        # Decoder: Reconstructs 64 latent features -> original input
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, input_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat, z

'''Autoencoder training function with Early Stopping'''
def train_autoencoder(X_train, X_test, DEVICE, verbose=True):
    model = AutoEncoder(X_train.shape[1]).to(DEVICE)
    early_stopper = EarlyStopping(patience=20) # Stop if no improvement for 20 epochs

    loader = DataLoader(
        torch.tensor(X_train, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()
    best_val = np.inf
    best_state = None
    train_losses, val_losses = [], []
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for xb in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            recon, _ = model(X_test_tensor)
            val_loss = criterion(recon, X_test_tensor).item()

        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        # Check Early Stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            if verbose: print(f"[AE] Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch+1) % 5 == 0: # Print every 5 epochs to reduce clutter
            print(f"[AE] Epoch {epoch+1:03d} | Train MSE: {epoch_loss:.5f} | Val MSE: {val_loss:.5f}")

    model.load_state_dict(best_state)
    return model, {"train_loss": train_losses, "val_loss": val_losses}

'''MLP classifier'''
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        # Adjusted architecture for smaller latent dim (64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5), # High dropout
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

'''Classifier training function'''
def train_classifier(X_train, y_train, X_test, y_test, n_classes, DEVICE, verbose=True):
    model = MLP(LATENT_DIM, n_classes).to(DEVICE)
    # patience for classifier can be higher as it fluctuates more
    early_stopper = EarlyStopping(patience=30) 

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam( # Adam often converges faster for classification
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}
    best_f1 = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        history["train_loss"].append(epoch_loss)

        model.eval()
        preds, trues = [], []
        val_loss_clf = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss_clf += criterion(out, yb).item() # Calculate val loss for early stopping
                preds.append(out.argmax(1).cpu().numpy())
                trues.append(yb.cpu().numpy())

        val_loss_clf /= len(val_loader)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")

        history["val_accuracy"].append(acc)
        history["val_f1"].append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
        
        # Early Stopping check on Validation Loss (not accuracy, as loss is smoother)
        early_stopper(val_loss_clf)
        if early_stopper.early_stop:
            if verbose: print(f"[CLF] Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch+1) % 5 == 0:
            print(f"[CLF] Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Macro-F1: {f1:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def visualize_latent_space(ae_model, X_data, y_labels, device):
    """
    Visualizes the latent space of the Autoencoder using t-SNE.
    
    Args:
        ae_model: The trained Autoencoder model
        X_data: Input features (e.g., x_test_scaled)
        y_labels: Class labels (e.g., y_test)
        device: 'cuda' or 'cpu'
    """
    print("Extracting latent features...")
    ae_model.eval()
    with torch.no_grad():
        # Convert to tensor and move to device
        inputs = torch.tensor(X_data, dtype=torch.float32).to(device)
        # Get the latent representation (z) from the encoder
        _, z_features = ae_model(inputs)
        z_features = z_features.cpu().numpy()

    print(f"Running t-SNE on {z_features.shape[0]} samples with {z_features.shape[1]} dimensions...")
    # Reduce to 2D for plotting
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_embedded = tsne.fit_transform(z_features)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=z_embedded[:, 0], 
        y=z_embedded[:, 1], 
        hue=y_labels, 
        palette="tab10", # Good color palette for distinct classes
        s=60,            # Marker size
        alpha=0.8        # Transparency
    )
    plt.title("t-SNE Visualization of Autoencoder Latent Space (dim=64)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


'''Denoising Autoencoder Model'''
class DenoisingAutoEncoder(nn.Module):
    def __init__(self, input_features, latent_features=LATENT_DIM):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, latent_features) # Latent Z
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, input_features)
        )

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat, z
    


'''DAE Training Function (Injects Noise)'''
def train_denoising_autoencoder(X_train, X_test, DEVICE, verbose=True):
    model = DenoisingAutoEncoder(X_train.shape[1]).to(DEVICE)
    early_stopper = EarlyStopping(patience=25) 

    loader = DataLoader(
        torch.tensor(X_train, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()
    best_val = np.inf
    best_state = None
    train_losses, val_losses = [], []
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for xb in loader:
            xb = xb.to(DEVICE)
            
            # --- NOISE INJECTION (The DAE Magic) ---
            # Create noise tensor same shape as input
            noise = torch.randn_like(xb) * NOISE_FACTOR
            # Add noise to input (clamp to avoid extreme values if necessary, generally not needed for StandardScaler)
            noisy_xb = xb + noise
            noisy_xb = noisy_xb.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass using NOISY input
            recon, _ = model(noisy_xb)
            
            # Compute loss against CLEAN original input (xb)
            loss = criterion(recon, xb) 
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        train_losses.append(epoch_loss)

        # Validation (No noise added during validation/testing)
        model.eval()
        with torch.no_grad():
            recon, _ = model(X_test_tensor)
            val_loss = criterion(recon, X_test_tensor).item()

        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        early_stopper(val_loss)
        if early_stopper.early_stop:
            if verbose: print(f"[DAE] Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch+1) % 5 == 0:
            print(f"[DAE] Epoch {epoch+1:03d} | Train MSE: {epoch_loss:.5f} | Val MSE: {val_loss:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, {"train_loss": train_losses, "val_loss": val_losses}




'''Variational Autoencoder (VAE) Model'''
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_features, latent_features=LATENT_DIM):
        super().__init__()
        
        # Shared Encoder Layers
        # We stop one layer before the bottleneck to split into mu and logvar
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE)
        )
        
        # Probabilistic Latent Space layers
        self.fc_mu = nn.Linear(512, latent_features)       # Mean vector
        self.fc_logvar = nn.Linear(512, latent_features)   # Log-Variance vector

        # Decoder (Standard reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(1024, input_features)
        )

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick:
        Allows backpropagation through random sampling.
        z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar) # Convert log-var to standard deviation
            eps = torch.randn_like(std)   # Sample random noise (epsilon)
            return mu + eps * std
        else:
            return mu # During testing, just return the mean (no noise)

    def forward(self, x):
        # Pass through shared encoder
        encoded = self.encoder(x)
        
        # Split into mean and variance
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        
        # Sample latent vector z
        z = self.reparameterize(mu, logvar)
        
        # Decode z back to x
        y_hat = self.decoder(z)
        
        return y_hat, z, mu, logvar

'''VAE Loss Function'''
def vae_loss_function(recon_x, x, mu, logvar, beta=0.001):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    """
    # 1. Reconstruction Loss (MSE) - how well did we restore the input?
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')

    # 2. KL Divergence - how far is our distribution from a standard Normal?
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # We take the mean to keep scale similar to MSE
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total Loss (beta weights the importance of KLD)
    # Small beta (e.g. 0.001) prioritizes reconstruction accuracy
    return MSE + (beta * KLD), MSE, KLD

'''VAE Training Function'''
def train_variational_autoencoder(X_train, X_test, DEVICE, verbose=True):
    model = VariationalAutoEncoder(X_train.shape[1]).to(DEVICE)
    early_stopper = EarlyStopping(patience=25)

    loader = DataLoader(
        torch.tensor(X_train, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val = np.inf
    best_state = None
    
    # Track metrics
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "train_kld": []}
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_kld = 0.0

        for xb in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            
            # VAE Forward pass returns 4 values
            recon, _, mu, logvar = model(xb)
            
            # Calculate VAE loss
            loss, mse, kld = vae_loss_function(recon, xb, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_kld += kld.item()

        # Average over batches
        epoch_loss /= len(loader)
        epoch_mse /= len(loader)
        epoch_kld /= len(loader)
        
        history["train_loss"].append(epoch_loss)
        history["train_mse"].append(epoch_mse)
        history["train_kld"].append(epoch_kld)

        # Validation
        model.eval()
        with torch.no_grad():
            recon, _, mu, logvar = model(X_test_tensor)
            val_loss, _, _ = vae_loss_function(recon, X_test_tensor, mu, logvar)
            val_loss = val_loss.item()

        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        early_stopper(val_loss)
        if early_stopper.early_stop:
            if verbose: print(f"[VAE] Early stopping at epoch {epoch+1}")
            break

        if verbose and (epoch+1) % 5 == 0:
            print(f"[VAE] Epoch {epoch+1:03d} | Loss: {epoch_loss:.5f} (MSE: {epoch_mse:.5f} KLD: {epoch_kld:.5f}) | Val: {val_loss:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, history