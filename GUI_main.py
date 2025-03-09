import tkinter as tk
import math
import cv2
import numpy as np
import random
import DrawingApp
import WordGeneratorApp

def main():
    root0 = tk.Tk()
    app = WordGeneratorApp.WordGeneratorApp(root0)
    root0.mainloop()
    chosen_word = app.chosen_word
    if not chosen_word:
        chosen_word = "No word"

    root = tk.Tk()
    app = DrawingApp.DrawingApp(root,chosen_word)
    root.mainloop()

if __name__ == "__main__":
    main()
