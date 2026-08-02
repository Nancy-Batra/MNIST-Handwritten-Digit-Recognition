import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import cv2
import numpy as np
from keras.models import load_model

model = load_model("best_mnist_model.keras")

#gui
root = tk.Tk()
root.title("Handwritten Digit Recognition")
root.geometry("500x600")
root.resizable(False, False)

image_label = tk.Label(root)
image_label.pack(pady=15)

result_label = tk.Label(
    root,
    text="Upload a digit image",
    font=("Arial",16)
)
result_label.pack()


#image preprocessing func
def preprocess_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    contours, _ = cv2.findContours( thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    if len(contours) == 0:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    digit = thresh[y:y+h, x:x+w]
    size = max(w, h)

    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w)//2
    y_offset = (size - h)//2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = digit

    # resizing to MNIST size
    square = cv2.resize(square, (28,28))

    # Normalize
    square = square.astype("float32")/255.0
    square = square.reshape(1,28,28,1)
    return square

def predict():
    path = filedialog.askopenfilename(
        filetypes=[ ("Images","*.png *.jpg *.jpeg")]
    )

    if not path:
        return

    # Show uploaded image
    img = Image.open(path)
    preview = img.resize((250,250))
    photo = ImageTk.PhotoImage(preview)
    image_label.config(image=photo)
    image_label.image = photo
    processed = preprocess_image(path)

    if processed is None:
        result_label.config(text="No digit detected")
        return

    prediction = model.predict(processed, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)*100

    result_label.config( text=f"Predicted Digit : {digit}\nConfidence : {confidence:.2f}%" )


button = tk.Button(
    root,
    text="Upload Image",
    command=predict,
    font=("Arial",14)
)

button.pack(pady=20)
root.mainloop()