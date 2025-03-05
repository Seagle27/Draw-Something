import tkinter as tk
import random

class WordGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Random Word Generator")
        master.configure(bg="lightblue")

        # List of words to choose from
        self.words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "kiwi", "lemon", "mango"]

        # Define fancy fonts for the labels
        fancy_font_title = ("Bernard MT Condensed", 48, "bold")
        fancy_font_word = ("Bauhaus 93", 72, "bold")

        # Label for the title text "Your word is:" without its own background
        self.title_label = tk.Label(master, text="Your word is:", font=fancy_font_title,
                                    bg=master.cget("bg"), fg="coral")
        self.title_label.pack(pady=40)

        # Label for displaying the random word without its own background
        self.word_label = tk.Label(master, text="", font=fancy_font_word,
                                   bg=master.cget("bg"), fg="salmon")
        self.word_label.pack(pady=110)

        # Button to change the word
        self.change_button = tk.Button(master, text="Change Word", command=self.change_word,
                                       font=("David", 20), bg="white", fg="black")
        self.change_button.pack(pady=10)

        # Button to approve the word and exit the GUI
        self.approve_button = tk.Button(master, text="Approve Word", command=self.approve_word,
                                        font=("David", 20), bg="lightgreen", fg="black")
        self.approve_button.pack(pady=10)

        # Initialize with a random word
        self.change_word()

    def change_word(self):
        new_word = random.choice(self.words)
        self.word_label.config(text=new_word)

    def approve_word(self):
        approved_word = self.word_label.cget("text")
        print("Approved word:", approved_word)
        self.master.destroy()
