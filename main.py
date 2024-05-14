import cv2
import numpy as np
import os

# Load the pre-trained model
from keras.src.saving import load_model

model = load_model('model/creditcard_improved.keras')


def preprocess(img, debug=True):
    # Image dimensions
    height, width = img.shape[:2]

    # Manually crop the image to include only the region with the digits
    y1 = int(height * 0.55)
    y2 = int(height * 0.65)
    x1 = int(width * 0.14)
    x2 = int(width * 0.87)
    img = img[y1:y2, x1:x2].copy()

    # Convert RGB to Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold for binary image (black and white)
    ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    if debug:
        cv2.imwrite("debug/preprocessed.jpg", thresh)

    return thresh


# Split the image into individual digit images
def split_image(image, debug=True):
    digit_images = []
    digit_clusters = 4  # There are 4 clusters of digits
    digits_per_cluster = 4  # Each cluster has 4 digits
    cluster_width = image.shape[1]  // digit_clusters  # Width of each cluster
    digit_width = (cluster_width) // digits_per_cluster   # Width of each digit

    for cluster in range(digit_clusters):
        x_cluster_start = cluster * cluster_width
        for i in range(digits_per_cluster):
            x1 = x_cluster_start + i * digit_width
            x2 = x1 + digit_width
            digit_image = image[:, x1:x2]
            digit_image = cv2.resize(digit_image, (32, 32))
            digit_image = np.stack((digit_image,) * 3, axis=-1)  # Convert to 3 channels
            digit_image = digit_image / 255.0  # Normalize pixel values
            digit_images.append(digit_image)
            if debug:
                cv2.imwrite(f"debug/digit_{cluster * digits_per_cluster + i}.png", (digit_image * 255).astype(np.uint8))

    digit_images = np.array(digit_images)
    return digit_images


# Recognize digits using the model
def recognize_digits(digit_images):
    predictions = model.predict(digit_images)
    digits = [np.argmax(pred) for pred in predictions]
    return digits


def main(image_path):
    # Create debug directory if it doesn't exist
    if not os.path.exists("debug"):
        os.makedirs("debug")

    # Read the image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess(image)

    # Split and recognize digits
    digit_images = split_image(preprocessed_image)

    digits = recognize_digits(digit_images)
    card_number = ''.join(map(str, digits))
    print(f"Recognized Card Number: {card_number}")
if __name__ == "__main__":
    image_path = 'creditcard.png'  # Update this to your image path
    main(image_path)
