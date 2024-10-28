import numpy as np
import matplotlib.pyplot as plt

# Funciones de activación
def step_function(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Para estabilidad numérica
    return exp_x / np.sum(exp_x)

class Perceptron:
    def __init__(self, input_size, hidden_size=5):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01  # Pesos de entrada a oculta
        self.weights_hidden_output = np.random.randn(hidden_size, 2) * 0.01  # Pesos de oculta a salida
    
    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden)
        hidden_layer_output = step_function(hidden_layer_input)  # Usar función de activación escalón
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        return softmax(output_layer_input)

    def train(self, X, y, learning_rate=0.1, epochs=100):
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward pass
                hidden_layer_input = np.dot(X[i], self.weights_input_hidden)
                hidden_layer_output = step_function(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
                output = softmax(output_layer_input)

                # Error y retropropagación
                error = y[i] - output
                # Retropropagación
                d_output = error
                hidden_layer_error = np.dot(d_output, self.weights_hidden_output.T)
                d_hidden_layer = hidden_layer_error * step_function(hidden_layer_output)

                # Actualizar pesos
                self.weights_hidden_output += learning_rate * np.outer(hidden_layer_output, d_output)
                self.weights_input_hidden += learning_rate * np.outer(X[i], d_hidden_layer)

# Generar datos de entrenamiento
def generate_data(num_samples=100):
    X = []
    y = []

    for _ in range(num_samples):
        # Generar un círculo
        circle = np.zeros((10, 10))
        rr, cc = np.ogrid[:10, :10]
        center = (5, 5)
        radius = np.random.randint(1, 4)  # Cambiar el tamaño del círculo aleatoriamente
        mask = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2
        circle[mask] = 1
        X.append(circle.flatten())  # Aplanar la matriz
        y.append([0, 1])  # Etiqueta para círculo (probabilidad [0, 1])

        # Generar una línea
        line = np.zeros((10, 10))
        line[np.random.randint(0, 10), :] = 1  # Línea horizontal en una posición aleatoria
        X.append(line.flatten())  # Aplanar la matriz
        y.append([1, 0])  # Etiqueta para línea (probabilidad [1, 0])

    return np.array(X), np.array(y)

# Generar datos de entrenamiento
X_train, y_train = generate_data(100)

# Inicializar el perceptrón
perceptron = Perceptron(input_size=100)

# Entrenar el perceptrón
perceptron.train(X_train, y_train, learning_rate=0.1, epochs=200)

# Generar datos de prueba
def generate_test_data(num_samples=30):
    X_test = []
    y_test = []

    for _ in range(num_samples):
        # Círculos
        circle = np.zeros((10, 10))
        rr, cc = np.ogrid[:10, :10]
        center = (5, 5)
        radius = np.random.randint(1, 4)  # Cambiar el tamaño del círculo aleatoriamente
        mask = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2
        circle[mask] = 1
        X_test.append(circle.flatten())
        y_test.append([0, 1])  # Etiqueta para círculo

        # Líneas
        line = np.zeros((10, 10))
        line[np.random.randint(0, 10), :] = 1
        X_test.append(line.flatten())
        y_test.append([1, 0])  # Etiqueta para línea

    return np.array(X_test), np.array(y_test)

# Generar datos de prueba
X_test, y_test = generate_test_data(30)

# Evaluar el perceptrón
correct_predictions = 0
predictions = perceptron.predict(X_test)

# Contar aciertos por clase
correct_circles = 0
correct_lines = 0

# Contadores para falsos positivos y negativos
false_positives = 0
false_negatives = 0

for i in range(len(X_test)):
    predicted_class = np.argmax(predictions[i])
    actual_class = np.argmax(y_test[i])
    if predicted_class == actual_class:
        correct_predictions += 1
        if actual_class == 1:  # Círculo
            correct_circles += 1
        else:  # Línea
            correct_lines += 1
    else:
        if actual_class == 1:  # Círculo fue clasificado incorrectamente como línea
            false_negatives += 1
        else:  # Línea fue clasificada incorrectamente como círculo
            false_positives += 1

# Calcular precisión
accuracy = correct_predictions / len(X_test)

# Mostrar resultados de forma estructurada
print("**Resultados del Modelo**")
print(f"- **Precisión del Modelo**: {accuracy * 100:.2f}%")
print(f"- **Aciertos Totales**:")
print(f"  - Círculos: {correct_circles}")
print(f"  - Líneas: {correct_lines}")
print(f"- **Falsos Positivos**: {false_positives}")
print(f"- **Falsos Negativos**: {false_negatives}")

# Crear y mostrar la matriz de confusión
confusion_matrix = np.array([[correct_lines, false_positives],
                             [false_negatives, correct_circles]])

print("\n**Matriz de Confusión:**")
print(confusion_matrix)

# Visualizar algunas predicciones
def visualize_predictions(X, predictions):
    plt.figure(figsize=(10, 5))
    for i in range(len(X)):
        plt.subplot(4, 15, i + 1)
        plt.imshow(X[i].reshape(10, 10), cmap='gray')
        plt.title("C" if np.argmax(predictions[i]) == 1 else "L")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualizar algunas predicciones
visualize_predictions(X_test, predictions)
