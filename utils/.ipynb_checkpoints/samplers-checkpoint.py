# ==========================================
#  MATH FUNCTIONS CLASS 
# ==========================================
import numpy as np
import torch

class MathFunctions:
    def __init__(self, num_points=128):
        self.num_points = num_points
        self.x_std = np.linspace(-5, 5, num_points)
        self.x_pos = np.linspace(0.1, 10, num_points)
        self.x_nonzero = np.concatenate([np.linspace(-5, -0.1, num_points//2),
                                         np.linspace(0.1, 5, num_points - num_points//2)])

    def _get_base_function(self, name, x_std, x_pos, x_nonzero):
        funcs = {
            'sin': (x_std, np.sin(x_std)),
            'tg': (x_std, np.clip(np.tan(x_std), -10, 10)),
            'sgn': (x_std, np.sign(x_std)),
            'sigmoid': (x_std, 1 / (1 + np.exp(-x_std))),
            'relu': (x_std, np.maximum(0, x_std)),
            'log10': (x_pos, np.log10(x_pos)),
            'log2': (x_pos, np.log2(x_pos)),
            '1_over_x': (x_nonzero, 1 / x_nonzero),
            'exp': (x_std, np.exp(x_std)),
            'linear': (x_std, x_std),
            'quadratic': (x_std, x_std**2),
            'cubic': (x_std, x_std**3),
            'ax_b': (x_std, 2.5 * x_std + 1.5),
            'sin_1_over_x': (x_nonzero, np.sin(1 / x_nonzero)),
            'sin_sq': (x_std, np.sin(x_std)**2),
            'gaussian': (x_std, np.exp(-x_std**2)),
            
            # 1. Fala prostokątna (ostre krawędzie)
            'square_wave': (x_std, np.sign(np.sin(2 * x_std))),
            # 2. Sygnał tłumiony (łączenie trendu wykładniczego z oscylacją)
            'damped_oscillator': (x_std, np.exp(-0.5 * np.abs(x_std)) * np.sin(5 * x_std)),
            # 3. Wysokie i niskie częstotliwości
            'mixed_freq': (x_std, np.sin(x_std) + 0.5 * np.cos(10 * x_std)),
            # 4. Chirp (częstotliwość rosnąca w czasie, test na zdolność adaptacji sieci)
            'chirp': (x_std, np.sin(x_std**2)),
            # 5. Funkcja Sinc
            'sinc': (x_std, np.sinc(x_std / np.pi)), 
            # 6. Krok Heaviside'a (nieciągłość i stabilność wokół zera)
            'step': (x_std, np.where(x_std > 0, 1.0, 0.0)),
            # 7. Abs (Moduł) - ostre "V" w zerze
            'abs': (x_std, np.abs(x_std))
        }
        
        if name not in funcs:
            raise ValueError(f"Funkcja '{name}' nie jest zaimplementowana.")
            
        return funcs[name]

    def _normalize(self, y):
        y_min, y_max = np.min(y), np.max(y)
        if y_max - y_min < 1e-6:
            return np.zeros_like(y)
        return 2 * (y - y_min) / (y_max - y_min) - 1

    def get_function(self, name):
        x, y = self._get_base_function(name, self.x_std, self.x_pos, self.x_nonzero)
        return x, self._normalize(y)

    def get_dataset(self, name, num_samples=1000, mode='train', seed=42):
	    """
	    POPRAWIONA WERSJA:
	    Przyjmuje seed z zewnątrz, dzięki czemu każdy RUN (0, 1, 2) dostanie inne dane,
	    ale zachowujemy separację między train/val/test.
	    """
	    # Dodajemy duże odstępy (offsety), żeby train nigdy nie "spotkał" się z testem
	    if mode == 'train':
	        current_seed = seed + 0
	    elif mode == 'val':
	        current_seed = seed + 10000 
	    elif mode == 'test':
	        current_seed = seed + 20000
	    else:
	        raise ValueError(f"Nieznany tryb: {mode}")
	    
	    np.random.seed(current_seed)
	        
	    X_data = []
	    Y_data = []
	
	    # Pobieramy referencyjną dziedzinę x
	    ref_x, _ = self._get_base_function(name, self.x_std, self.x_pos, self.x_nonzero)
	        
	    for _ in range(num_samples):
	        x_shift = np.random.uniform(-1.0, 1.0)
	        a_scale = np.random.uniform(0.5, 1.5)
	            
	        if np.array_equal(ref_x, self.x_std):
	            x_shifted = self.x_std + x_shift
	            x_for_call = x_shifted
	        elif np.array_equal(ref_x, self.x_pos):
	            shift_pos = np.random.uniform(0.1, 2.0)
	            x_shifted = self.x_pos + shift_pos
	            x_for_call = x_shifted
	        else: 
	            x_shifted = self.x_nonzero
	            x_for_call = self.x_nonzero
	            
	        # Generowanie wartości y dla przesuniętego x
	        _, y = self._get_base_function(name, x_std=x_for_call, x_pos=x_for_call, x_nonzero=x_for_call)
	            
	        # Skalowanie amplitudy i normalizacja
	        y = a_scale * y
	        X_data.append(ref_x) 
	        Y_data.append(self._normalize(y))
	            
	    return np.array(X_data), np.array(Y_data)	    
    def generate_dataset(self, name, num_train=2000, num_val=400):
        """
        MOST MIĘDZY EKSPERYMENTAMI:
        Pobiera dane z get_dataset, ale przygotowuje je pod format 
        wymagany przez nowy ExperimentRunner.
        """
        # 1. Pobieramy dane (zwracają NumPy arrays)
        _, y_train = self.get_dataset(name, num_samples=num_train, mode='train')
        _, y_val = self.get_dataset(name, num_samples=num_val, mode='val')

        # 2. Konwersja na Tensory (rozwiązuje błąd TypeError: 'int' object is not callable)
        train_tensor = torch.from_numpy(y_train).float()
        val_tensor = torch.from_numpy(y_val).float()

        # 3. Dostosowanie kształtu (tylko dla potrzeb Eksperymentu II)
        # Modele Conv1D i UNet wymagają formatu (Batch, Channels, Length)
        # Unsqueeze dodaje brakujący wymiar kanału (Channels=1)
        if train_tensor.dim() == 2:
            train_tensor = train_tensor.unsqueeze(1)
        if val_tensor.dim() == 2:
            val_tensor = val_tensor.unsqueeze(1)

        return train_tensor, val_tensor

def sample_sin_batch(batch_size, num_points=128):
    x = np.linspace(-1, 1, num_points)
    batch = []
    for _ in range(batch_size):
        A = np.random.uniform(0.5, 2.0)
        f = np.random.uniform(1, 10)
        phi = np.random.uniform(0, 2*np.pi)
        y = A * np.sin(2 * np.pi * f * x + phi)
        batch.append(y)
    return torch.tensor(batch, dtype=torch.float32)

def sample_chirp_batch(batch_size, num_points=128):
    x = np.linspace(0, 1, num_points)
    batch = []
    for _ in range(batch_size):
        f0 = np.random.uniform(1, 5)
        f1 = np.random.uniform(10, 30)
        y = np.sin(2 * np.pi * (f0 + (f1 - f0) * x) * x)
        batch.append(y)
    return torch.tensor(batch, dtype=torch.float32)

def sample_hard_batch(batch_size, num_points=128):
    x = np.linspace(0, 1, num_points)
    batch = []

    for _ in range(batch_size):
        y = (
            0.7 * np.sin(2*np.pi*np.random.uniform(1, 3)*x) +
            0.3 * np.sin(2*np.pi*np.random.uniform(10, 25)*x)
        )

        f0 = np.random.uniform(2, 5)
        f1 = np.random.uniform(15, 40)
        y += 0.3 * np.sin(2*np.pi*(f0 + (f1 - f0)*x)*x)

        for _ in range(np.random.randint(1, 4)):
            center = np.random.randint(10, num_points-10)
            width = np.random.randint(3, 10)
            y[center-width:center+width] += np.random.uniform(-1.0, 1.0)

        envelope = 1 + 0.5 * np.sin(2*np.pi*np.random.uniform(0.5, 2)*x)
        y *= envelope

        batch.append(y)

    return torch.tensor(batch, dtype=torch.float32)