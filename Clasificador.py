import math
import random
from collections import defaultdict
from statistics import mean, stdev

# Función para calcular la distancia euclidiana
def euclidean_distancia(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Clasificador KNN
def clasificador_knn(caracteristicas, clases, punto_prueba, k=3):
    distancias = [(euclidean_distancia(punto, punto_prueba), clase) for punto, clase in zip(caracteristicas, clases)]
    distancias.sort(key=lambda x: x[0])
    k_vecinos = distancias[:k]
    clases_vecinos = [clase for _, clase in k_vecinos]
    return max(set(clases_vecinos), key=clases_vecinos.count)

# Clasificador Naive Bayes
def clasificador_naive_bayes(caracteristicas, clases, punto_prueba):
    clases_unicas = set(clases)
    prob_clases = {clase: clases.count(clase) / len(clases) for clase in clases_unicas}
    prob_condicional = defaultdict(lambda: defaultdict(list))

    for i, clase in enumerate(clases):
        for j, valor in enumerate(caracteristicas[i]):
            prob_condicional[clase][j].append(valor)

    for clase in clases_unicas:
        for j in range(len(caracteristicas[0])):
            valores = prob_condicional[clase][j]
            mean_val = sum(valores) / len(valores)
            stdev_val = math.sqrt(sum((x - mean_val) ** 2 for x in valores) / len(valores))
            prob_condicional[clase][j] = (mean_val, stdev_val)

    def probabilidad_gaussiana(x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    prob_posterior = {}
    for clase in clases_unicas:
        prob_posterior[clase] = prob_clases[clase]
        for j, valor in enumerate(punto_prueba):
            mean, stdev = prob_condicional[clase][j]
            prob_posterior[clase] *= probabilidad_gaussiana(valor, mean, stdev)

    return max(prob_posterior, key=prob_posterior.get)

###3 distanias media

def clasificador_euclidiano(caracteristicas, clases, punto_prueba):
    """Clasifica un punto de prueba utilizando la distancia media de cada clase."""
    
    # Diccionario para almacenar la suma de distancias y el conteo de elementos por clase
    distancia_por_clase = {}
    
    # Calcular la distancia de cada punto al punto de prueba y acumularla por clase
    for i in range(len(caracteristicas)):
        distancia = euclidean_distancia(caracteristicas[i], punto_prueba)
        clase_actual = clases[i]
        
        if clase_actual not in distancia_por_clase:
            distancia_por_clase[clase_actual] = {'suma_distancia': 0, 'conteo': 0}
        
        distancia_por_clase[clase_actual]['suma_distancia'] += distancia
        distancia_por_clase[clase_actual]['conteo'] += 1
    
    # Calcular la distancia media por clase
    distancia_media_por_clase = {
        clase: datos['suma_distancia'] / datos['conteo'] 
        for clase, datos in distancia_por_clase.items()
    }
    
    # Encontrar la clase con la distancia media más baja
    clase_predicha = min(distancia_media_por_clase, key=distancia_media_por_clase.get)
    
    return clase_predicha

def clasificador_1nn(caracteristicas, clases, punto_prueba):
    """Clasifica un punto de prueba utilizando el clasificador 1-NN."""
    
    # Inicializar la distancia mínima como infinito
    min_distance = float('inf')
    clase_determinada = None
    
    # Recorrer cada punto de entrenamiento
    for i in range(len(caracteristicas)):
        distancia = euclidean_distancia(caracteristicas[i], punto_prueba)
        
        # Si la distancia actual es menor que la mínima, actualizar
        if distancia < min_distance:
            min_distance = distancia
            clase_determinada = clases[i]
    
    return clase_determinada

# Función para calcular la matriz de confusión
def matriz_confusion(true_labels, pred_labels):
    cm = {}
    # Inicializa las claves de las clases si no existen
    for true, pred in zip(true_labels, pred_labels):
        if true not in cm:
            cm[true] = {}
        if pred not in cm[true]:
            cm[true][pred] = 0
        cm[true][pred] += 1
    return cm

# Función para calcular el Accuracy
def accuracy(valores_verdaderos, valores_predecir):
    """Calcula el accuracy (precisión) del clasificador."""
    correct = sum([1 for verdaderos, test in zip(valores_verdaderos, valores_predecir) if verdaderos == test])
    return correct / len(valores_verdaderos)

# Función para dividir el conjunto de datos en Hold Out 70/30
def hold_out_70_30(caracteristicas, clases, clasificador, test_size=0.3):
    """Validación Hold Out 70/30"""
    datacombinada = list(zip(caracteristicas, clases))
    random.shuffle(datacombinada)
    split_cantidad_datos = int(len(datacombinada) * (1 - test_size))
    entrenamiento_data = datacombinada[:split_cantidad_datos]
    test_data = datacombinada[split_cantidad_datos:]
    
    entrenamiento_features, entrenamiento_labels = zip(*entrenamiento_data)
    test_features, test_labels = zip(*test_data)
    
    # Predecir
    y_pred = [clasificador(entrenamiento_features, entrenamiento_labels, x) for x in test_features]
    
    # Calcular desempeño
    acc = accuracy(test_labels, y_pred)
    cm = matriz_confusion(test_labels, y_pred)
    return acc, cm

import random

def k_fold_cross_validation(caracteristicas, clases, clasificador, k=10):
    """Validación 10-Fold Cross-Validation"""
    # Combina las características y clases en una lista de tuplas
    datacombinada = list(zip(caracteristicas, clases))
    
    # Mezcla aleatoriamente los datos
    random.shuffle(datacombinada)
    
    # Calcula el tamaño de cada pliegue (fold)
    fold_size = len(datacombinada) // k
    
    accuracies = []  # Para almacenar las precisiones de cada pliegue
    confusion_matrices = []  # Para almacenar las matrices de confusión de cada pliegue
    
    for i in range(k):
        # Divide en k pliegues: test y train
        test_data = datacombinada[i * fold_size : (i + 1) * fold_size]
        train_data = datacombinada[:i * fold_size] + datacombinada[(i + 1) * fold_size:]
        
        # Separa las características y las etiquetas para entrenamiento y prueba
        entrenamiento_features, entrenamiento_labels = zip(*train_data)
        test_features, test_labels = zip(*test_data)
        
        # Predecir
        y_pred = [clasificador(entrenamiento_features, entrenamiento_labels, x) for x in test_features]
        
        # Calcular desempeño
        acc = accuracy(test_labels, y_pred)
        cm = matriz_confusion(test_labels, y_pred)
        
        # Almacena el desempeño de este pliegue
        accuracies.append(acc)
        confusion_matrices.append(cm)
    
    # Promedio de las precisiones
    avg_acc = sum(accuracies) / len(accuracies)
    
    # Promedio de las matrices de confusión
    avg_cm = {}
    for cm in confusion_matrices:
        for true_class in cm:
            for pred_class in cm[true_class]:
                if (true_class, pred_class) not in avg_cm:
                    avg_cm[(true_class, pred_class)] = 0
                avg_cm[(true_class, pred_class)] += cm[true_class][pred_class]
    
    return avg_acc, avg_cm

import random

# Función para Leave-One-Out
def leave_one_out(caracteristicas, clases, clasificador):
    """Validación Leave-One-Out"""
    # Combina las características y clases en una lista de tuplas
    datacombinada = list(zip(caracteristicas, clases))
    
    accuracies = []  # Para almacenar las precisiones de cada iteración
    confusion_matrices = []  # Para almacenar las matrices de confusión de cada iteración
    
    for i in range(len(datacombinada)):
        # Divide en datos de prueba (un solo dato) y datos de entrenamiento (el resto)
        test_data = [datacombinada[i]]
        train_data = datacombinada[:i] + datacombinada[i+1:]
        
        # Separa las características y las etiquetas para entrenamiento y prueba
        entrenamiento_features, entrenamiento_labels = zip(*train_data)
        test_features, test_labels = zip(*test_data)
        
        # Predecir
        y_pred = [clasificador(entrenamiento_features, entrenamiento_labels, x) for x in test_features]
        
        # Calcular desempeño
        acc = accuracy(test_labels, y_pred)
        cm = matriz_confusion(test_labels, y_pred)
        
        # Almacena el desempeño de esta iteración
        accuracies.append(acc)
        confusion_matrices.append(cm)
    
    # Promedio de las precisiones
    avg_acc = sum(accuracies) / len(accuracies)
    
    # Promedio de las matrices de confusión
    avg_cm = {}
    for cm in confusion_matrices:
        for true_class in cm:
            for pred_class in cm[true_class]:
                if (true_class, pred_class) not in avg_cm:
                    avg_cm[(true_class, pred_class)] = 0
                avg_cm[(true_class, pred_class)] += cm[true_class][pred_class]
    
    return avg_acc, avg_cm