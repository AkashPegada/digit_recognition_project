import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load your trained model
model = load_model("digit_recognizer.h5")

# Initialize canvas + drawing
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.chart_widget = None

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_btn = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_btn.grid(row=0, column=0)

        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=1)

        self.result_label = tk.Label(root, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.result_label.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (280, 280), "white")
        self.draw_obj = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw_obj = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit and click Predict")

    def predict(self):
        # Resize, invert, normalize
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        preds = model.predict(img)[0]
        top_3 = preds.argsort()[-3:][::-1]

        result_text = "Top 3 Predictions:\n"
        for i in top_3:
            result_text += f"Digit: {i}, Confidence: {preds[i]*100:.2f}%\n"

        self.result_label.config(text=result_text)
        # Plot confidence scores
        self.plot_confidence(preds)

    def plot_confidence(self, preds):
        # Destroy previous chart widget if it exists
        if self.chart_widget:
            self.chart_widget.get_tk_widget().destroy()
            self.chart_widget = None

        # Create fresh figure
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.bar(range(10), preds, tick_label=list(range(10)))
        ax.set_title("Prediction Confidence")
        ax.set_ylim([0, 1])

        # Embed the chart into Tkinter
        self.chart_widget = FigureCanvasTkAgg(fig, master=self.root)
        self.chart_widget.draw()
        self.chart_widget.get_tk_widget().pack()
        
        # Explicitly close the figure to avoid memory leaks
        plt.close(fig)


# Launch app
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
    print("App closed. Goodbye")