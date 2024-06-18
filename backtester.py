from keras import models
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

generator = models.load_model("generator-mnist-cgan.h5")

def model_check(model, plot=1, num_test_images=1000):

    rows = 10
    cols = 10
    
    max_plots = 100
    
    num_samples = num_test_images
    num_labels = 10
    noise_dim = 100
    
    noises = tf.random.normal([num_samples, noise_dim])

    test_labels = np.random.choice(num_labels, size=(num_samples, 1))
    label = tf.convert_to_tensor(test_labels, dtype=tf.int32)
    
    generated_imgs = generator([noises, label], training=False)
    
    predictions = model.predict(generated_imgs)
    
    correctly_predicted = 0
    incorrectly_predicted = 0
    
    if plot:
        plt.figure(figsize=(rows * 1, cols * 1))
        print(f"Model tested on ={num_samples} images.")
        print(f"Plotting top {min(max_plots,num_samples)} predicted and ground truth results.")
    
    for i in range(num_test_images):
        prediction = np.argmax(predictions[i])

        image = (generated_imgs[i, :, :, 0].numpy() * 127.5 + 127.5).astype("uint8")
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
        image = cv2.merge([image] * 3)
        
        if prediction == test_labels[i]:
            correctly_predicted += 1
            rgb_color = (0, 255, 0)  # green for correct predictions
        else:
            incorrectly_predicted += 1
            rgb_color = (255, 0, 0)  # red for wrong predictions

        cv2.putText(image, str(prediction), (0, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, rgb_color, 1)

        if plot and i < max_plots:
            plt.subplot(rows, cols, i + 1, title="label: %s" % test_labels[i])
            plt.axis('Off')  
            plt.imshow(image)
                
    if plot:
        plt.tight_layout()
        plt.show()
    

    accuracy = (correctly_predicted / num_test_images) * 100
    print(f"Number of correctly predicted samples: {correctly_predicted}")
    print(f"Number of incorrectly predicted samples: {incorrectly_predicted}")
    print(f"Accuracy of the model: {accuracy:.2f}%")

    return