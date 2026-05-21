import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time


# -------- Synthetic 2D Shapes Dataset --------
class Shapes2DDataset(Dataset):
    def __init__(self, samples_per_shape=1000, img_size=64):
        self.img_size = img_size
        self.shapes = ['square', 'circle', 'triangle']
        self.x_vals = np.linspace(-1, 1, 32)
        self.y_vals = np.linspace(-1, 1, 32)
        self.scale_vals = np.linspace(0.3, 0.9, 6)
        self.rot_vals = np.linspace(0, 2*np.pi, 40, endpoint=False)
        self.factor_names = ['shape', 'posX', 'posY', 'scale', 'rotation']
        self.factor_sizes = [3, 32, 32, 6, 40]

        all_factors = []
        for shape_idx, shape in enumerate(self.shapes):
            for x in self.x_vals:
                for y in self.y_vals:
                    for s in self.scale_vals:
                        for r in self.rot_vals:
                            all_factors.append((shape_idx, x, y, s, r))
        np.random.seed(42)
        indices = np.random.choice(len(all_factors), samples_per_shape, replace=False)
        self.factors = [all_factors[i] for i in indices]

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):
        shape_idx, x, y, scale, rot = self.factors[idx]
        img = self._draw_shape(shape_idx, x, y, scale, rot)
        factors = torch.tensor([shape_idx, x, y, scale, rot], dtype=torch.float32)
        return img.unsqueeze(0), factors

    def _draw_shape(self, shape_idx, x, y, scale, rot):
        img = torch.zeros(self.img_size, self.img_size)
        xs = torch.linspace(-1, 1, self.img_size)
        ys = torch.linspace(-1, 1, self.img_size)
        xx, yy = torch.meshgrid(xs, ys, indexing='ij')
        xt = xx - x
        yt = yy - y
        cos_r = np.cos(rot)
        sin_r = np.sin(rot)
        xr = cos_r * xt + sin_r * yt
        yr = -sin_r * xt + cos_r * yt
        xs = xr / scale
        ys = yr / scale

        if self.shapes[shape_idx] == 'square':
            mask = (torch.abs(xs) <= 1) & (torch.abs(ys) <= 1)
        elif self.shapes[shape_idx] == 'circle':
            mask = (xs**2 + ys**2) <= 1
        elif self.shapes[shape_idx] == 'triangle':
            v1x, v1y = 0.0, -1.0
            v2x, v2y = -np.sqrt(3)/2, 0.5
            v3x, v3y = np.sqrt(3)/2, 0.5
            def sign(p1x, p1y, p2x, p2y, p3x, p3y):
                return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
            d1 = sign(xs, ys, v1x, v1y, v2x, v2y)
            d2 = sign(xs, ys, v2x, v2y, v3x, v3y)
            d3 = sign(xs, ys, v3x, v3y, v1x, v1y)
            has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
            has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
            mask = ~(has_neg & has_pos)
        else:
            raise ValueError("Unknown shape")
        img[mask] = 1.0
        return img


# -------- β-VAE Model --------
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 4, 4)          # исправлено: 512 каналов
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# -------- loss function --------
def beta_vae_loss(recon_x, x, mu, logvar, beta_norm=1.0, latent_dim=10, img_size=64):
    BCE = F.binary_cross_entropy(recon_x.view(x.size(0), -1), x.view(x.size(0), -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    beta_abs = beta_norm * img_size * img_size / latent_dim
    return BCE + beta_abs * KLD, BCE, KLD


# -------- metric helpers --------
def compute_disentanglement_metric(model, dataset, device, num_train=5000, num_eval=1000, batch_size=64):
    model.eval()
    factor_sizes = dataset.factor_sizes
    K = len(factor_sizes)
    np.random.seed(123)

    def indices_to_values(factor_indices):
        shape_idx = factor_indices[0]
        posX = dataset.x_vals[factor_indices[1]]
        posY = dataset.y_vals[factor_indices[2]]
        scale = dataset.scale_vals[factor_indices[3]]
        rot = dataset.rot_vals[factor_indices[4]]
        return shape_idx, posX, posY, scale, rot

    train_data, train_labels = [], []
    for _ in range(num_train):
        y = np.random.randint(0, K)
        f1 = [np.random.randint(0, s) for s in factor_sizes]
        f2 = [np.random.randint(0, s) for s in factor_sizes]
        f2[y] = f1[y]
        v1 = indices_to_values(f1)
        v2 = indices_to_values(f2)
        img1 = dataset._draw_shape(*v1).unsqueeze(0).unsqueeze(0).to(device)
        img2 = dataset._draw_shape(*v2).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            mu1, _ = model.encode(img1)
            mu2, _ = model.encode(img2)
        diff = torch.abs(mu1 - mu2).cpu().numpy().flatten()
        train_data.append(diff)
        train_labels.append(y)

    X_train = torch.tensor(np.array(train_data), dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train = (X_train - mean) / std

    classifier = nn.Linear(model.latent_dim, K).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    classifier.train()
    for epoch in range(20):
        perm = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx].to(device)
            yb = y_train[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(xb), yb)
            loss.backward()
            optimizer.step()

    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(num_eval):
            y = np.random.randint(0, K)
            f1 = [np.random.randint(0, s) for s in factor_sizes]
            f2 = [np.random.randint(0, s) for s in factor_sizes]
            f2[y] = f1[y]
            v1 = indices_to_values(f1)
            v2 = indices_to_values(f2)
            img1 = dataset._draw_shape(*v1).unsqueeze(0).unsqueeze(0).to(device)
            img2 = dataset._draw_shape(*v2).unsqueeze(0).unsqueeze(0).to(device)
            mu1, _ = model.encode(img1)
            mu2, _ = model.encode(img2)
            diff = torch.abs(mu1 - mu2).cpu().numpy().flatten()
            diff_tensor = (torch.tensor(diff, dtype=torch.float32) - mean.squeeze()) / std.squeeze()
            logits = classifier(diff_tensor.unsqueeze(0).to(device))
            if torch.argmax(logits).item() == y:
                correct += 1
            total += 1
    return correct / total


def compute_mse(model, loader, device):
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            recon, _, _ = model(data)
            total_mse += F.mse_loss(recon, data, reduction='sum').item()
    return total_mse / len(loader.dataset)


# -------- training function --------
def train_beta_vae(model, train_loader, optimizer, beta_norm, device, latent_dim=10):
    model.train()
    total_loss, total_bce, total_kld = 0.0, 0.0, 0.0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss, bce, kld = beta_vae_loss(recon, data, mu, logvar, beta_norm, latent_dim=latent_dim)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
    n = len(train_loader.dataset)
    return total_loss / n, total_bce / n, total_kld / n


# -------- experiment runner --------
def run_beta_experiment(beta_norm, device, train_loader, val_loader, dataset_for_metric, epochs=200):
    model = BetaVAE(latent_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = train_beta_vae(model, train_loader, optimizer, beta_norm, device, latent_dim=10)
        if epoch % 10 == 0:
            print(f"β_norm={beta_norm} Epoch {epoch:3d}: Loss={train_loss:.4f}")
    mse = compute_mse(model, val_loader, device)
    disent_score = compute_disentanglement_metric(model, dataset_for_metric, device)
    return {"beta_norm": beta_norm, "mse": round(mse, 6), "disentanglement_accuracy": round(disent_score, 4)}


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = Shapes2DDataset(samples_per_shape=10000)   # увеличенный датасет
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                       generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    beta_norms = [0.5, 1, 2, 4, 8, 16, 32, 64]   # нормализованные β
    results = []
    start_time = time.time()

    for beta_norm in beta_norms:
        res = run_beta_experiment(beta_norm, device, train_loader, val_loader, full_dataset, epochs=200)
        results.append(res)

    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time/60:.1f} min")

    output = {"results": results, "total_time_min": round(total_time/60, 1)}
    with open("beta_vae_experiment_results_2.json", "w") as f:
        json.dump(output, f, indent=2)

    for r in results:
        print(f"β_norm={r['beta_norm']:5.1f}  MSE={r['mse']:.6f}  DisentAcc={r['disentanglement_accuracy']:.4f}")