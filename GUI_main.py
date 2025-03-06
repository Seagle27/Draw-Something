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
    root = tk.Tk()
    app = DrawingApp.DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
