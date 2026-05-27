import os
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflowjs as tfjs

def create_model():
    """
    Cria a arquitetura simples da CNN conforme solicitado.
    - 1 Camada de Convolução (4 filtros, kernel 3x3)
    - 1 Camada de Ativação (ReLU)
    - 1 Camada de Max Pooling (2x2)
    - 1 Camada Densa (10 classes para o MNIST)
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1), name='input_1'),
        layers.Conv2D(4, (3, 3), padding='same', name='conv_1'),
        layers.Activation('relu', name='relu_1'),
        layers.MaxPooling2D((2, 2), name='pool_1'),
        layers.Flatten(name='flatten_1'),
        layers.Dense(10, activation='softmax', name='dense_1')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data():
    """Carrega e normaliza o dataset MNIST."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalizar pixels para o intervalo [0, 1] e ajustar formato
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    return (train_images, train_labels), (test_images, test_labels)

def main():
    print("Criando o modelo...")
    model = create_model()
    model.summary()
    
    print("\nCarregando dataset MNIST...")
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    print("\nIniciando treinamento (1 época para fins didáticos)...")
    # Para o clone didático, 1 época já demonstra pesos mudando, mas vamos usar 3
    model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))
    
    export_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'model')
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"\nExportando modelo para TensorFlow.js em: {export_dir}")
    # Exporta para a pasta public do frontend para ser carregado localmente pelo Vite
    tfjs.converters.save_keras_model(model, export_dir)
    print("Exportação concluída com sucesso!")

if __name__ == '__main__':
    main()
