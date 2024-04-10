import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import webbrowser

class GIFDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GIF Display Project")

        # Display GIF button
        self.display_button = tk.Button(root, text="Display GIF", command=self.display_gif)
        self.display_button.pack()

        # Open YouTube button
        self.youtube_button = tk.Button(root, text="Open YouTube", command=self.open_youtube)
        self.youtube_button.pack()

        self.window = tk.Toplevel(self.root)
        self.window.title("GIF Display")

        self.canvas = tk.Canvas(self.window, bg="white")
        self.canvas.pack()

    def display_gif(self):
        file_path = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if file_path:
            gif = Image.open(file_path)
            gif = gif.resize((200, 200), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(gif)

            # Update the canvas with the new image
            self.canvas.config(width=gif.width, height=gif.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.image = self.photo

    def open_youtube(self):
        webbrowser.open("https://www.youtube.com")

if __name__ == "__main__":
    root = tk.Tk()
    app = GIFDisplayApp(root)
    root.mainloop()