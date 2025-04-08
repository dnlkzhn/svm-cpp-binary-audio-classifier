import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from predict import predict_genre

class AIcoustIQApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Konfigurácia hlavného okna
        self.title("AIcoustIQ")
        self.geometry("800x600")  # Default size

        # Nastavenie vzhľadu a témy customtkinter
        ctk.set_appearance_mode("dark")  
        ctk.set_default_color_theme("blue")

        # Konfigurácia rozloženia mriežky pre škálovateľnosť
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Vytvorenie komponentov používateľského rozhrania
        self.create_widgets()

    def create_widgets(self):
        """Creates and places all UI widgets inside the main window."""

        # Hlavný rám na umiestnenie všetkých prvkov používateľského rozhrania
        self.frame = ctk.CTkFrame(self, fg_color="#302f2f")
        self.frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Rozšíriteľný rám
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Konfigurácia rozloženia mriežky v rámci (tri stĺpce)
        self.frame.grid_columnconfigure(0, weight=1)  
        self.frame.grid_columnconfigure(1, weight=2)  
        self.frame.grid_columnconfigure(2, weight=1)  

        for i in range(6): 
            self.frame.grid_rowconfigure(i, weight=1)

        # Widget na zadávanie súborov (na celú šírku)
        self.file_entry = ctk.CTkEntry(self.frame, height=50, font=("Arial", 14))
        self.file_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Štítok na zobrazenie výsledku predpovede
        self.result_label = ctk.CTkLabel(self.frame, text="", font=("Arial", 34))
        self.result_label.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Spoločný štýl tlačidiel
        button_style = {
            "font": ("Arial", 22),
            "width": 50,
            "height": 50,
            "corner_radius": 15,
            "fg_color": "#3A3B3C",
            "hover_color": "#5A5B5C",
            "border_width": 2,
            "border_color": "white",
            "text_color": "white"
        }

        # Umiestnenie všetkých tlačidiel v strednom stĺpci
        self.browse_button = ctk.CTkButton(self.frame, text="Browse", command=self.browse_file, **button_style)
        self.browse_button.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

        self.predict_button = ctk.CTkButton(self.frame, text="Predict Genre", command=self.classify_audio, **button_style)
        self.predict_button.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

        self.clear_button = ctk.CTkButton(self.frame, text="Clear", command=self.clear_result, **button_style)
        self.clear_button.grid(row=4, column=1, padx=20, pady=10, sticky="ew")

        self.guide_button = ctk.CTkButton(self.frame, text="Show Instructions", command=self.show_instructions, **button_style)
        self.guide_button.grid(row=5, column=1, padx=20, pady=10, sticky="ew")

        # Štítok na zobrazenie krátkej používateľskej príručky
        self.guide_label = ctk.CTkLabel(self.frame, text="", font=("Arial", 18), wraplength=600)
        self.guide_label.grid(row=6, column=1, padx=10, pady=10, sticky="ew")


    def browse_file(self):
        """Opens file dialog to select an audio file and updates the file entry."""
        filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if filename:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)

    def classify_audio(self):
        """Runs the genre prediction function on the selected file."""
        audio_path = self.file_entry.get()

        # Validacia cesty k súboru
        if not os.path.exists(audio_path):
            messagebox.showerror("Error", "Invalid file path. Please select a valid WAV file.")
            return

        # Cesty pre dataset a model 
        dataset_path = r""
        model_path = r""

        try:
            # Volanie funkcie predikcie
            genre = predict_genre(audio_path, model_path, dataset_path)
            self.result_label.configure(text=f"Predicted Genre: {genre}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

    def clear_result(self):
        """Clears the prediction result label and file entry field."""
        self.result_label.configure(text="")  
        self.guide_label.configure(text="")  
        self.file_entry.delete(0, "end")  

    def show_instructions(self):
        """Shows instructions for the user."""
        self.guide_label.configure(text="Welcome to AIcoustIQ. First, provide a file by either entering its path manually in the input field at the top of the screen or by clicking the Browse button to select the file. The file must be in .wav format. Then, click Predict to run the identification algorithm. Once the process is complete, the predicted genre will appear in the middle of the screen.")

# Run the application
if __name__ == "__main__":
    app = AIcoustIQApp()
    app.mainloop()
