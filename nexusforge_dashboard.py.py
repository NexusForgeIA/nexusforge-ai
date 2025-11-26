import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_resource
def load_model():
    return QuantumCor3(n_lags=3, epochs=500)

class QuantumCor3:
    def __init__(self, n_lags=5, sample_size=1000, lr=0.001, epochs=1000):
        self.scaler = self._SimpleScaler()
        self.model = None
        self.n_lags = n_lags
        self.sample_size = sample_size
        self.lr = lr
        self.epochs = epochs
        self._lags_applied = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.intensity = 4.5

    class _SimpleScaler:
        def __init__(self): self.mean_ = None; self.scale_ = None
        def fit_transform(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0) + 1e-8
            return (X - self.mean_) / self.scale_
        def transform(self, X):
            return (X - self.mean_) / self.scale_

    class _SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32]):
            super().__init__()
            layers = []
            prev = input_size
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x): return self.net(x)

    def _create_lags(self, X, y):
        n_samples, n_features = X.shape
        X_lagged = np.zeros((n_samples - self.n_lags, n_features * (self.n_lags + 1)))
        y_lagged = y[self.n_lags:]
        for i in range(self.n_lags + 1):
            X_lagged[:, i*n_features:(i+1)*n_features] = X[i : i + n_samples - self.n_lags]
        return X_lagged, y_lagged

    def _holographic_correlation(self, X):
        n = X.shape[0]
        if n > self.sample_size:
            idx = np.random.choice(n, self.sample_size, replace=False)
            idx_sort = np.sort(idx)
            X_s = X[idx]
            dist = squareform(pdist(X_s, 'euclidean'))
            corr = np.exp(-dist * self.intensity)
            proj = corr @ X_s
            projected = np.zeros_like(X)
            for j in range(X.shape[1]):
                projected[:, j] = np.interp(np.arange(n), idx_sort, proj[:, j])
            return projected
        else:
            dist = squareform(pdist(X, 'euclidean'))
            return np.exp(-dist * self.intensity) @ X

    def _train_step(self, X_mapped, y, epochs=None):
        if epochs is None: epochs = self.epochs // 5
        model = self._SimpleMLP(X_mapped.shape[1]).to(self.device)
        dataset = TensorDataset(torch.tensor(X_mapped, dtype=torch.float32),
                                torch.tensor(y.reshape(-1,1), dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
        return model

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame): X = X.values
        y = np.array(y)
        if X.shape[1] > 1 and len(y.shape) == 1:
            X, y = self._create_lags(X, y)
            self._lags_applied = True
        X_scaled = self.scaler.fit_transform(X)
        X_mapped = self._holographic_correlation(X_scaled)
        split = int(0.8 * len(X_mapped))
        if split > 10:
            temp = self._train_step(X_mapped[:split], y[:split])
            with torch.no_grad():
                pred = temp(torch.tensor(X_mapped[split:], dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
            st.info(f"CV R² ≈ {r2_score(y[split:], pred):.4f}")
        self.model = self._train_step(X_mapped, y, self.epochs)
        st.success("¡Modelo entrenado con éxito!")

    def predict(self, X, optimize=True):
        if isinstance(X, pd.DataFrame): X = X.values
        X_scaled = self.scaler.transform(X)
        X_mapped = self._holographic_correlation(X_scaled)
        self.model.eval()
        with torch.no_grad():
            raw = self.model(torch.tensor(X_mapped, dtype=torch.float32).to(self.device)).cpu().numpy().flatten()
        if not optimize: return raw
        gamma = 0.14
        vol = np.std(raw) / 10
        opt = raw * (1 - gamma * np.tanh(np.abs(raw - np.median(raw)) / vol))
        return {"prediction": opt.tolist(), "risk_delta": float(np.max(np.abs(raw - opt))), "optimized": True}

    def audit(self, X, y_real):
        pred = self.predict(X, optimize=False)
        mse = mean_squared_error(y_real, pred)
        r2 = r2_score(y_real, pred)
        quality = "HIGH" if mse < 0.05 else "MEDIUM" if mse < 0.2 else "LOW"
        return {"mse": float(mse), "r2": float(r2), "quality": quality}

# ==================== DASHBOARD ====================
st.title("NexusForge AI Dashboard")
st.image("logo_nexusforge.png", width=450, caption="NexusForge AI – Forjando Predicciones IBEX")
st.markdown("**Predicciones élite para proyectos IBEX 35**")

st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube CSV de IBEX", type="csv")
n_lags = st.sidebar.slider("Lags históricos", 1, 10, 3)
epochs = st.sidebar.slider("Épocas", 100, 1000, 500)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados:", data.head())

    target_col = st.selectbox("Columna objetivo", data.columns)
    feature_cols = st.multiselect("Features", [c for c in data.columns if c != target_col],
                                  default=[c for c in data.columns if c != target_col])

    if st.button("Entrenar Modelo"):
        with st.spinner("Entrenando..."):
            model = load_model()
            model.n_lags = n_lags
            model.epochs = epochs
            model.fit(data[feature_cols], data[target_col])
            st.session_state.model = model

    if 'model' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Predecir últimas 5"):
                pred = st.session_state.model.predict(data[feature_cols].tail(5))
                st.json(pred)
                st.bar_chart(pred["prediction"])
        with col2:
            if st.button("Auditar"):
                split = int(0.8 * len(data))
                audit = st.session_state.model.audit(data[feature_cols].iloc[split:], data[target_col].iloc[split:])
                st.json(audit)
                st.metric("Calidad", audit["quality"])

else:
    st.info("Sube un CSV o usa el demo")
    if st.button("Datos Demo IBEX"):
        np.random.seed(42)
        dates = pd.date_range('2025-01-01', periods=100, freq='B')
        demo = pd.DataFrame(np.cumsum(np.random.randn(100,5)*0.02 + 0.001, axis=0) + 11000,
                            index=dates, columns=['Open','High','Low','Volume','RSI'])
        demo['Close'] = demo['High'].shift(-1).fillna(demo['High'])
        st.dataframe(demo.head())
        st.download_button("Descargar demo", demo.to_csv(), "ibex_demo.csv")