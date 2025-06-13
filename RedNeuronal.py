import h5py
import numpy as np
from scipy.optimize import minimize

class ActivationFunction:
    """Clase base para funciones de activación"""
    def activate(self, z):
        raise NotImplementedError
        
    def derivative(self, a, z=None):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    def activate(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):  # Simplificar la función
        a = self.activate(z)
        return a * (1 - a)

class ReLU(ActivationFunction):
    def activate(self, z):
        return np.maximum(0, z)
    
    def derivative(self, a=None, z=None):
        if z is not None:
            return (z > 0).astype(float)
        elif a is not None:
            return (a > 0).astype(float)
        else:
            raise ValueError("Either a or z must be provided")

class Softmax(ActivationFunction):
    def activate(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)
    
    def derivative(self, a=None, z=None):
        # Derivada no se usa en backpropagation para softmax
        return np.ones(1)

class Tanh(ActivationFunction):
    def activate(self, z):
        return np.tanh(z)
    
    def derivative(self, a, z=None):
        return 1 - a**2

class RedNeuronal:
    def __init__(self, output_activation=Softmax(), hidden_activation=ReLU(), hidden_layers=1, lambda_=0.01):
        self.X = None
        self.y = None
        self.lambda_ = lambda_
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        
        if hidden_layers not in [1, 2]:
            raise ValueError("Solo se soportan 1 o 2 capas ocultas")
        
        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        self.theta1 = None
        self.theta2 = None
        self.theta3 = None

    def configurar_capas(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def inicializar_parametros(self):
        # Inicialización capa de entrada -> oculta
        if isinstance(self.hidden_activation, ReLU):
            epsilon1 = np.sqrt(2.0 / self.input_size)
        else:
            epsilon1 = np.sqrt(1.0 / self.input_size)
        
        self.theta1 = np.random.randn(self.hidden_size, self.input_size + 1) * epsilon1
        
        if self.hidden_layers == 1:
            if isinstance(self.output_activation, ReLU):
                epsilon2 = np.sqrt(2.0 / self.hidden_size)
            else:
                epsilon2 = np.sqrt(1.0 / self.hidden_size)
            self.theta2 = np.random.randn(self.output_size, self.hidden_size + 1) * epsilon2
        else:
            if isinstance(self.hidden_activation, ReLU):
                epsilon2 = np.sqrt(2.0 / self.hidden_size)
            else:
                epsilon2 = np.sqrt(1.0 / self.hidden_size)
            self.theta2 = np.random.randn(self.hidden_size, self.hidden_size + 1) * epsilon2
            
            # Corrección importante: Inicialización correcta para capa de salida
            if isinstance(self.output_activation, ReLU):
                epsilon3 = np.sqrt(2.0 / self.hidden_size)
            else:
                epsilon3 = np.sqrt(1.0 / self.hidden_size)
            self.theta3 = np.random.randn(self.output_size, self.hidden_size + 1) * epsilon3

    def fit(self, x, y):
        self.X = x
        self.y = y

    def _forward_propagation(self, X, thetas):
        m = X.shape[0]
        a1 = np.c_[np.ones(m), X]
        
        z2 = a1 @ thetas[0].T
        a2 = self.hidden_activation.activate(z2)
        a2 = np.c_[np.ones(m), a2]
        
        if len(thetas) == 2:
            z3 = a2 @ thetas[1].T
            a3 = self.output_activation.activate(z3)
            return a1, z2, a2, z3, a3, None, None
        else:
            z3 = a2 @ thetas[1].T
            a3 = self.hidden_activation.activate(z3)
            a3 = np.c_[np.ones(m), a3]
            z4 = a3 @ thetas[2].T
            a4 = self.output_activation.activate(z4)
            return a1, z2, a2, z3, a3, z4, a4

    def funcion_costo_gradiente(self, t):
        # Reconstruir parámetros
        if self.hidden_layers == 1:
            t1 = t[:self.theta1.size].reshape(self.theta1.shape)
            t2 = t[self.theta1.size:].reshape(self.theta2.shape)
            thetas = [t1, t2]
        else:
            t1 = t[:self.theta1.size].reshape(self.theta1.shape)
            t2 = t[self.theta1.size:self.theta1.size+self.theta2.size].reshape(self.theta2.shape)
            t3 = t[self.theta1.size+self.theta2.size:].reshape(self.theta3.shape)
            thetas = [t1, t2, t3]
        
        m = self.X.shape[0]
        a1, z2, a2, z3, a3, z4, a4 = self._forward_propagation(self.X, thetas)
        y_vec = np.eye(self.output_size)[self.y.reshape(-1)]
        
        # Calcular función de costo
        if self.hidden_layers == 1:
            h = a3
        else:
            h = a4

        # CORRECCIÓN IMPORTANTE: Cálculo correcto de costo
        if isinstance(self.output_activation, Softmax):
            j = -np.sum(y_vec * np.log(h + 1e-8)) / m
        else:
            j = -np.sum(y_vec * np.log(h + 1e-8) + (1 - y_vec) * np.log(1 - h + 1e-8)) / m
        
        # CORRECCIÓN IMPORTANTE: Regularización para todas las capas
        reg_term = 0
        for theta in thetas:
            reg_term += np.sum(theta[:, 1:]**2)  # Excluir bias
        j += (self.lambda_ / (2 * m)) * reg_term

        # Backpropagation
        if self.hidden_layers == 1:
            # CORRECCIÓN: Manejo correcto de gradiente para Softmax
            if isinstance(self.output_activation, Softmax):
                delta3 = (a3 - y_vec)
            else:
                delta3 = (a3 - y_vec) * self.output_activation.derivative(a3)
            
            delta2 = (delta3 @ thetas[1][:, 1:]) * self.hidden_activation.derivative(z2)
            
            grad2 = delta3.T @ a2 / m
            grad1 = delta2.T @ a1 / m
            
            grad1[:, 1:] += (self.lambda_ / m) * thetas[0][:, 1:]
            grad2[:, 1:] += (self.lambda_ / m) * thetas[1][:, 1:]
            
            return j, np.concatenate([grad1.ravel(), grad2.ravel()])
        
        else:
            # CORRECCIÓN: Manejo correcto de gradiente para Softmax
            if isinstance(self.output_activation, Softmax):
                delta4 = (a4 - y_vec)
            else:
                delta4 = (a4 - y_vec) * self.output_activation.derivative(a4)
            
            delta3 = (delta4 @ thetas[2][:, 1:]) * self.hidden_activation.derivative(z3)
            delta2 = (delta3 @ thetas[1][:, 1:]) * self.hidden_activation.derivative(z2)
            
            grad3 = delta4.T @ a3 / m
            grad2 = delta3.T @ a2 / m
            grad1 = delta2.T @ a1 / m
            
            grad1[:, 1:] += (self.lambda_ / m) * thetas[0][:, 1:]
            grad2[:, 1:] += (self.lambda_ / m) * thetas[1][:, 1:]
            grad3[:, 1:] += (self.lambda_ / m) * thetas[2][:, 1:]
            
            return j, np.concatenate([grad1.ravel(), grad2.ravel(), grad3.ravel()])

    def entrenar(self, destino=None, callback=None, max_iter=100, paciencia=5):
        if self.hidden_layers == 1:
            theta_inicial = np.concatenate([self.theta1.ravel(), self.theta2.ravel()])
        else:
            theta_inicial = np.concatenate([self.theta1.ravel(), self.theta2.ravel(), self.theta3.ravel()])
        
        # Implementación de Early Stopping
        mejor_costo = np.inf
        mejor_theta = theta_inicial.copy()
        cuenta_espera = 0
        
        def callback_mejorada(theta):
            nonlocal mejor_costo, cuenta_espera, mejor_theta
            j, _ = self.funcion_costo_gradiente(theta)
            if j < mejor_costo:
                mejor_costo = j
                mejor_theta = theta.copy()
                cuenta_espera = 0
            else:
                cuenta_espera += 1
            
            if callback:
                callback(theta)
                
            return cuenta_espera < paciencia
        
        opciones = {'maxiter': max_iter}
        res = minimize(self.funcion_costo_gradiente, theta_inicial, 
                       jac=True, method='L-BFGS-B', options=opciones, 
                       callback=callback_mejorada)
        
        # Usar los mejores parámetros encontrados
        res.x = mejor_theta
        
        if destino:
            self._guardar_modelo(res.x, destino)
        return res

    def _guardar_modelo(self, theta_optimo, destino):
        if self.hidden_layers == 1:
            t1_size = self.capa2 * (self.capa1 + 1)
            self.theta1 = np.reshape(theta_optimo[:t1_size], (self.capa2, (self.capa1 + 1)))
            self.theta2 = np.reshape(theta_optimo[t1_size:], (self.capa3, (self.capa2 + 1)))
        else:
            t1_size = self.capa2 * (self.capa1 + 1)
            t2_size = self.capa3 * (self.capa2 + 1)
            self.theta1 = np.reshape(theta_optimo[:t1_size], (self.capa2, (self.capa1 + 1)))
            self.theta2 = np.reshape(theta_optimo[t1_size:t1_size+t2_size], (self.capa3, (self.capa2 + 1)))
            self.theta3 = np.reshape(theta_optimo[t1_size+t2_size:], (self.capa4, (self.capa3 + 1)))
        
        with h5py.File(destino, 'w') as arch:
            arch.create_dataset('Theta1', data=self.theta1)
            arch.create_dataset('Theta2', data=self.theta2)
            if self.hidden_layers == 2:
                arch.create_dataset('Theta3', data=self.theta3)

    def cargar(self, archivo):
        with h5py.File(archivo, 'r') as arch:
            self.theta1 = arch['Theta1'][:]
            self.theta2 = arch['Theta2'][:]
            if 'Theta3' in arch:
                self.theta3 = arch['Theta3'][:]
                self.hidden_layers = 2
            else:
                self.hidden_layers = 1

    def _predict_forward(self, X):
        m = X.shape[0]
        a1 = np.c_[np.ones(m), X]
        
        z2 = a1 @ self.theta1.T
        a2 = self.hidden_activation.activate(z2)
        a2 = np.c_[np.ones(m), a2]
        
        if self.hidden_layers == 1:
            z3 = a2 @ self.theta2.T
            return self.output_activation.activate(z3)
        else:
            z3 = a2 @ self.theta2.T
            a3 = self.hidden_activation.activate(z3)
            a3 = np.c_[np.ones(m), a3]
            z4 = a3 @ self.theta3.T
            return self.output_activation.activate(z4)

    def predecir_clase(self, X):
        h = self._predict_forward(X)
        return np.argmax(h, axis=1)

    def predecir_probabilidad(self, X):
        return self._predict_forward(X)

    def predecir(self, X):
        prob = self._predict_forward(X)
        clases = np.argmax(prob, axis=1)
        probabilidades = prob[np.arange(len(prob)), clases]
        return clases, probabilidades

    # Métodos para evaluación de modelos
    def matriz_confusion(self, X, y_real):
        y_pred = self.predecir_clase(X)
        n_clases = self.output_size
        return self._calcular_matriz_confusion(y_real, y_pred, n_clases)
    
    def _calcular_matriz_confusion(self, y_real, y_pred, n_clases):
        matriz = np.zeros((n_clases, n_clases), dtype=int)
        for i in range(len(y_real)):
            real = int(y_real[i])
            pred = int(y_pred[i])
            matriz[real, pred] += 1
        return matriz

    def accuracy(self, matriz):
        return np.trace(matriz) / np.sum(matriz)

    def precision_por_clase(self, matriz):
        precisiones = np.zeros(matriz.shape[0])
        for i in range(matriz.shape[0]):
            if np.sum(matriz[:, i]) > 0:
                precisiones[i] = matriz[i, i] / np.sum(matriz[:, i])
        return precisiones

    def recall_por_clase(self, matriz):
        recalls = np.zeros(matriz.shape[0])
        for i in range(matriz.shape[0]):
            if np.sum(matriz[i, :]) > 0:
                recalls[i] = matriz[i, i] / np.sum(matriz[i, :])
        return recalls

    def f1_score_por_clase(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall + 1e-8)
    
    def metricas_completas(self, X, y_real):
        matriz = self.matriz_confusion(X, y_real)
        acc = self.accuracy(matriz)
        precision = self.precision_por_clase(matriz)
        recall = self.recall_por_clase(matriz)
        f1 = self.f1_score_por_clase(precision, recall)
        
        return {
            'matriz_confusion': matriz,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }