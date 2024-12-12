import tkinter as tk
from tkinter import messagebox, simpledialog, Listbox, Scrollbar, ttk
import os
import sounddevice as sd
import wave
import librosa
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import soundfile as sf
import json
from vosk import Model, KaldiRecognizer
from datetime import datetime
import noisereduce as nr
import logging
import time
import threading

# Initialize logging
logging.basicConfig(
    filename='voice_auth_log.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize the encoder and Vosk model
logging.info("Initializing VoiceEncoder and Vosk model")
encoder = VoiceEncoder()
vosk_model = Model("vosk-model-small-en-us-0.15")  # Ensure you have downloaded a Vosk model into the 'model' directory
logging.info("Initialization complete")

# Function to extract speaker embedding
def extract_speaker_embedding(audio_path):
    logging.debug(f"Extracting speaker embedding from '{audio_path}'")
    wav = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(wav)
    logging.debug("Speaker embedding extracted successfully")
    return embedding

# Function to add noise to audio
def add_noise(audio, noise_factor=0.005):
    logging.debug("Adding noise to audio")
    noise = np.random.randn(len(audio))
    audio_with_noise = audio + noise_factor * noise
    audio_with_noise = np.clip(audio_with_noise, -1.0, 1.0)
    logging.debug("Noise added successfully")
    return audio_with_noise

# Function to reduce noise from audio
def reduce_noise(audio, rate):
    logging.debug("Reducing noise from audio")
    noise_profile = audio[0:rate // 2]
    reduced_audio = nr.reduce_noise(y=audio, sr=rate, y_noise=noise_profile, prop_decrease=0.8)
    logging.debug("Noise reduced successfully")
    return reduced_audio

# Function to check if the keyword matches
def check_keyword(audio_path, expected_keywords):
    logging.debug(f"Checking keyword in audio '{audio_path}'")
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(True)

    keyword_detected = False
    confidence_threshold = 0.8  # Set a confidence threshold

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "text" in result:
                detected_text = result["text"].lower()
                for keyword in expected_keywords:
                    if keyword in detected_text:
                        logging.info(f"Keyword '{keyword}' detected with text '{detected_text}'")
                        if result.get("confidence", 1.0) >= confidence_threshold:
                            logging.info(f"Keyword '{keyword}' passed confidence threshold.")
                            keyword_detected = True
                            break
        if keyword_detected:
            break

    wf.close()
    if not keyword_detected:
        logging.warning(f"No matching keyword detected or confidence too low among {expected_keywords}")
    return keyword_detected

# Tkinter Application Class
class VoiceAuthApp:
    def __init__(self, master):
        self.root = master
        self.root.title("Voice Authentication System")

        # Define instance attributes here
        self.username_entry = None
        self.keyword_entry = None
        self.verify_username_entry = None
        self.threshold_entry = None
        self.similarity_threshold = 0.8  # Default similarity threshold
        self.progress_bar = None
        self.create_welcome_screen()

    def create_welcome_screen(self):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Welcome Label
        welcome_label = tk.Label(self.root, text="Welcome to Voice Authentication", font=("Helvetica", 16))
        welcome_label.pack(pady=20)

        # Threshold Label and Entry
        threshold_label = tk.Label(self.root, text="Set Similarity Threshold (0-1):", font=("Helvetica", 12))
        threshold_label.pack(pady=5)
        self.threshold_entry = tk.Entry(self.root, font=("Helvetica", 12))
        self.threshold_entry.insert(0, str(self.similarity_threshold))
        self.threshold_entry.pack(pady=5)

        # Update Threshold Button
        update_threshold_button = tk.Button(self.root, text="Update Threshold", command=self.update_threshold, width=20, height=2)
        update_threshold_button.pack(pady=10)

        # Register Button
        register_button = tk.Button(self.root, text="Register", command=self.create_register_screen, width=20, height=2)
        register_button.pack(pady=10)

        # Verify Button
        verify_button = tk.Button(self.root, text="Verify", command=self.create_verify_screen, width=20, height=2)
        verify_button.pack(pady=10)

        # Manage Users Button
        manage_users_button = tk.Button(self.root, text="Manage Users", command=self.create_user_management_screen, width=20, height=2)
        manage_users_button.pack(pady=10)

    def update_threshold(self):
        try:
            new_threshold = float(self.threshold_entry.get())
            if 0 <= new_threshold <= 1:
                self.similarity_threshold = new_threshold
                logging.info(f"Similarity threshold updated to {new_threshold:.2f}")
                messagebox.showinfo("Success", f"Similarity threshold updated to {new_threshold:.2f}.")
            else:
                logging.error("Threshold update failed: Value out of range (0-1)")
                messagebox.showerror("Input Error", "Threshold must be between 0 and 1.")
        except ValueError:
            logging.error("Threshold update failed: Invalid input")
            messagebox.showerror("Input Error", "Please enter a valid number between 0 and 1.")

    def create_register_screen(self):
        # Clear widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Username Entry
        username_label = tk.Label(self.root, text="Enter Username:", font=("Helvetica", 12))
        username_label.pack(pady=5)
        self.username_entry = tk.Entry(self.root, font=("Helvetica", 12))
        self.username_entry.pack(pady=5)

        # Custom Keyword Entry
        keyword_label = tk.Label(self.root, text="Enter Custom Keyword:", font=("Helvetica", 12))
        keyword_label.pack(pady=5)
        self.keyword_entry = tk.Entry(self.root, font=("Helvetica", 12))
        self.keyword_entry.pack(pady=5)

        # Progress Bar for Registration
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Register Button
        register_button = tk.Button(self.root, text="Start Recording", command=self.register_with_countdown, width=20, height=2)
        register_button.pack(pady=10)

        # Back Button
        back_button = tk.Button(self.root, text="Back", command=self.create_welcome_screen, width=20, height=2)
        back_button.pack(pady=10)

    def create_verify_screen(self):
        # Clear widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Username Entry
        username_label = tk.Label(self.root, text="Enter Username:", font=("Helvetica", 12))
        username_label.pack(pady=5)
        self.verify_username_entry = tk.Entry(self.root, font=("Helvetica", 12))
        self.verify_username_entry.pack(pady=5)

        # Progress Bar for Verification
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Verify Button
        verify_button = tk.Button(self.root, text="Start Verification", command=self.verify_with_countdown, width=20, height=2)
        verify_button.pack(pady=10)

        # Back Button
        back_button = tk.Button(self.root, text="Back", command=self.create_welcome_screen, width=20, height=2)
        back_button.pack(pady=10)

    def register_with_countdown(self):
        # Countdown Timer Label
        countdown_label = tk.Label(self.root, text="Get ready! Registration starts in 3 seconds", font=("Helvetica", 12))
        countdown_label.pack(pady=10)

        # Start Countdown
        self.start_countdown(countdown_label, 3, self.prepare_recording_ui)

    def verify_with_countdown(self):
        # Countdown Timer Label
        countdown_label = tk.Label(self.root, text="Get ready! Verification starts in 3 seconds", font=("Helvetica", 12))
        countdown_label.pack(pady=10)

        # Start Countdown
        self.start_countdown(countdown_label, 3, self.prepare_verification_ui)

    def start_countdown(self, label, seconds, next_action):
        if seconds >= 0:
            label.config(text=f"Get ready! Starting in {seconds} seconds")
            self.root.after(1000, self.start_countdown, label, seconds - 1, next_action)
        else:
            label.destroy()
            next_action()

    def prepare_recording_ui(self):
        self.display_recording_ui("Recording in progress...")
        self.root.after(500, self.register_user)  # Slight delay to ensure UI updates

    def prepare_verification_ui(self):
        self.display_recording_ui("Verification in progress...")
        self.root.after(500, self.verify_user)  # Slight delay to ensure UI updates

    def display_recording_ui(self, message):
        # Recording Label
        recording_label = tk.Label(self.root, text=message, font=("Helvetica", 12), fg="red")
        recording_label.pack(pady=10)

        # Moving Circle Animation
        canvas = tk.Canvas(self.root, width=100, height=100)
        canvas.pack()
        circle = canvas.create_oval(10, 10, 90, 90, fill="red")

        def animate_circle():
            if "oval" in canvas.gettags(circle):
                canvas.move(circle, 5, 0)  # Move the circle
                coords = canvas.coords(circle)
                if coords[2] >= 100 or coords[0] <= 0:
                    canvas.move(circle, -5, 0)
                self.root.after(100, animate_circle)

        animate_circle()

    def beep_sound(self):
        try:
            import os
            os.system('say "Recording started"')  # MacOS/Linux
        except Exception:
            logging.warning("Beep sound not supported on this platform.")

    def register_user(self):
        self.beep_sound()  # Beep sound when recording starts
        username = self.username_entry.get().strip().lower()
        custom_keyword = self.keyword_entry.get().strip().lower()

        if not username or not custom_keyword:
            logging.error("Registration failed: Username or keyword not provided")
            messagebox.showerror("Input Error", "Please enter both a username and a custom keyword.")
            return

        try:
            logging.info(f"Starting registration for user '{username}' with keyword '{custom_keyword}'")

            # Timer Label
            timer_label = tk.Label(self.root, text="Recording... Elapsed Time: 0 seconds", font=("Helvetica", 12))
            timer_label.pack(pady=5)
            self.update_elapsed_timer(timer_label, 0)

            self.update_progress(0)
            time.sleep(0.5)  # Simulating progress
            self._register_user_logic(username, custom_keyword)
            self.update_progress(100)

            logging.info(f"User '{username}' registered successfully with keyword '{custom_keyword}'")
            messagebox.showinfo("Success", f"User '{username}' registered successfully with keyword '{custom_keyword}'.")
            self.create_welcome_screen()
        except Exception as e:
            logging.error(f"Error during registration for user '{username}': {str(e)}")
            messagebox.showerror("Error", str(e))


    def verify_user(self):
        username = self.verify_username_entry.get().strip().lower()

        if not username:
            logging.error("Verification failed: Username not provided")
            messagebox.showerror("Input Error", "Please enter a username.")
            return

        try:
            logging.info(f"Starting verification for user '{username}'")
            self.update_progress(0)

            timer_label = tk.Label(self.root, text="Recording... Elapsed Time: 0 seconds", font=("Helvetica", 12))
            timer_label.pack(pady=5)
            self.update_elapsed_timer(timer_label, 0)

            time.sleep(0.5)  # Simulating progress

            # Verify Keyword and Similarity
            verification_folder = f"data/verification/{username}"
            embedding_folder = f"data/embeddings/{username}"
            registered_embedding_path = f"{embedding_folder}/{username}_embedding.npy"
            keyword_path = f"{embedding_folder}/{username}_keyword.txt"
            audio_path_prefix = f"{verification_folder}/{username}_verification"

            if not os.path.exists(registered_embedding_path) or not os.path.exists(keyword_path):
                logging.error(f"User '{username}' not found for verification")
                raise ValueError(f"User '{username}' not found!")

            with open(keyword_path, "r") as f:
                custom_keyword = f.read().strip().lower()

            logging.info(f"Recording audio for verification of user '{username}'")
            audio_path = self.record_audio(audio_path_prefix)
            logging.debug(f"Audio recorded for verification of user '{username}' at '{audio_path}'")

            detected_keyword = check_keyword(audio_path, expected_keywords=[custom_keyword])
            if not detected_keyword:
                messagebox.showerror("Failed", f"Keyword not detected or incorrect. Expected: '{custom_keyword}'.")
                logging.warning(f"Verification failed for user '{username}': Incorrect keyword")
                return

            # Compute Similarity
            registered_embedding = np.load(registered_embedding_path)
            new_embedding = extract_speaker_embedding(audio_path)
            similarity = 1 - cosine(registered_embedding, new_embedding)
            logging.info(f"Similarity score for user '{username}': {similarity:.4f}")

            if similarity > self.similarity_threshold:
                message = (
                    f"Congratulations!\n"
                    f"User: {username}\n"
                    f"Keyword: {detected_keyword}\n"
                    f"Similarity: {similarity:.2f}\n"
                    "Voice verified successfully!"
                )
                messagebox.showinfo("Success", message)
                logging.info(f"User '{username}' verified successfully")
            else:
                message = (
                    f"Verification Failed!\n"
                    f"User: {username}\n"
                    f"Keyword: {detected_keyword}\n"
                    f"Similarity: {similarity:.2f}\n"
                    "Voice verification failed. Please try again."
                )
                messagebox.showerror("Failed", message)
                logging.warning(f"Verification failed for user '{username}'")
            self.create_welcome_screen()
        except Exception as e:
            logging.error(f"Error during verification for user '{username}': {str(e)}")
            messagebox.showerror("Error", str(e))

    def update_elapsed_timer(self, label, elapsed_seconds):
        if label.winfo_exists():
            label.config(text=f"Recording... Elapsed Time: {elapsed_seconds} seconds")
            self.root.after(1000, self.update_elapsed_timer, label, elapsed_seconds + 1)

    def update_progress(self, value):
        if self.progress_bar:
            self.progress_bar["value"] = value
        self.root.update_idletasks()

    def _register_user_logic(self, username, custom_keyword):
        registration_folder = f"data/registration/{username}"
        embedding_folder = f"data/embeddings/{username}"
        audio_path_prefix = f"{registration_folder}/{username}_registration"
        embedding_path = f"{embedding_folder}/{username}_embedding.npy"
        keyword_path = f"{embedding_folder}/{username}_keyword.txt"

        os.makedirs(registration_folder, exist_ok=True)
        os.makedirs(embedding_folder, exist_ok=True)

        logging.info(f"Recording audio for registration of user '{username}'")
        audio_path = self.record_audio(audio_path_prefix)
        logging.debug(f"Audio recorded for user '{username}' at '{audio_path}'")

        audio, rate = librosa.load(audio_path, sr=16000)
        embeddings = []

        # Original Embedding
        logging.info(f"Extracting original embedding for user '{username}'")
        original_embedding = extract_speaker_embedding(audio_path)
        embeddings.append(original_embedding)
        logging.debug(f"Original embedding extracted for user '{username}'")

        # Augmented Embeddings with Noise
        for i in range(3):
            logging.info(f"Adding noise and extracting embedding {i + 1} for user '{username}'")
            audio_with_noise = add_noise(audio)
            noisy_audio_path = f"{registration_folder}/{username}_registration_noisy_{i}.wav"
            sf.write(noisy_audio_path, audio_with_noise, rate)
            logging.debug(f"Noisy audio saved for user '{username}' at '{noisy_audio_path}'")
            noisy_embedding = extract_speaker_embedding(noisy_audio_path)
            embeddings.append(noisy_embedding)
            logging.debug(f"Noisy embedding {i + 1} extracted for user '{username}'")

        final_embedding = np.mean(embeddings, axis=0)
        np.save(embedding_path, final_embedding)
        logging.info(f"Final embedding saved to '{embedding_path}' for user '{username}'")

        with open(keyword_path, "w") as f:
            f.write(custom_keyword)
        logging.info(f"Keyword '{custom_keyword}' saved to '{keyword_path}' for user '{username}'")

    def _verify_user_logic(self, username):
        verification_folder = f"data/verification/{username}"
        embedding_folder = f"data/embeddings/{username}"
        registered_embedding_path = f"{embedding_folder}/{username}_embedding.npy"
        keyword_path = f"{embedding_folder}/{username}_keyword.txt"
        audio_path_prefix = f"{verification_folder}/{username}_verification"

        if not os.path.exists(registered_embedding_path) or not os.path.exists(keyword_path):
            logging.error(f"User '{username}' not found for verification")
            raise ValueError(f"User '{username}' not found!")

        os.makedirs(verification_folder, exist_ok=True)

        with open(keyword_path, "r") as f:
            custom_keyword = f.read().strip().lower()

        logging.info(f"Recording audio for verification of user '{username}'")
        audio_path = self.record_audio(audio_path_prefix)
        logging.debug(f"Audio recorded for verification of user '{username}' at '{audio_path}'")

        if not check_keyword(audio_path, expected_keywords=[custom_keyword]):
            logging.warning(f"Verification failed for user '{username}': Incorrect keyword")
            return False

        logging.info(f"Extracting embedding for verification of user '{username}'")
        registered_embedding = np.load(registered_embedding_path)
        new_embedding = extract_speaker_embedding(audio_path)

        similarity = 1 - cosine(registered_embedding, new_embedding)
        logging.info(f"Similarity score for user '{username}': {similarity:.4f}")
        return similarity > self.similarity_threshold

    @staticmethod
    def record_audio(filename_prefix, duration=3, fs=16000):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.wav"

        if os.path.exists(filename):
            os.remove(filename)

        logging.info(f"Recording audio to '{filename}'")
        print(f"Recording... Please say the keyword clearly...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
        sd.wait()

        if np.all(audio == 0):
            logging.error("Recording failed or captured silence. Please check the microphone.")
            raise ValueError("Recording failed or captured silence. Please check the microphone.")

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes(audio.tobytes())
        logging.info(f"Audio recorded and saved to '{filename}'")
        return filename

    def create_user_management_screen(self):
        # Clear existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # User List Label
        user_list_label = tk.Label(self.root, text="Registered Users:", font=("Helvetica", 12))
        user_list_label.pack(pady=5)

        # Listbox for Users
        self.user_listbox = Listbox(self.root, font=("Helvetica", 12), width=30, height=10)
        self.user_listbox.pack(pady=5)

        # Scrollbar for the listbox
        scrollbar = Scrollbar(self.root)
        self.user_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar.config(command=self.user_listbox.yview)

        # Load users into listbox
        self.load_users()

        # Delete User Button
        delete_button = tk.Button(self.root, text="Delete User", command=self.delete_user, width=20, height=2)
        delete_button.pack(pady=10)

        # Update User Button
        update_button = tk.Button(self.root, text="Update User", command=self.update_user, width=20, height=2)
        update_button.pack(pady=10)

        # Back Button
        back_button = tk.Button(self.root, text="Back", command=self.create_welcome_screen, width=20, height=2)
        back_button.pack(pady=10)

    def load_users(self):
        self.user_listbox.delete(0, tk.END)
        if os.path.exists("data/embeddings"):
            for user in os.listdir("data/embeddings"):
                if user != ".DS_Store":  # Skip .DS_Store
                    self.user_listbox.insert(tk.END, user)

    def delete_user(self):
        selected_user = self.user_listbox.get(tk.ACTIVE)

        if not selected_user:
            logging.error("Deletion failed: No user selected")
            messagebox.showerror("Error", "Please select a user to delete.")
            return

        confirmation = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete user '{selected_user}'?")
        if confirmation:
            logging.info(f"Deleting user '{selected_user}'")
            embedding_folder = f"data/embeddings/{selected_user}"
            registration_folder = f"data/registration/{selected_user}"
            verification_folder = f"data/verification/{selected_user}"

            # Remove user data
            if os.path.exists(embedding_folder):
                os.system(f"rm -rf {embedding_folder}")
                logging.debug(f"Deleted embedding folder for user '{selected_user}'")
            if os.path.exists(registration_folder):
                os.system(f"rm -rf {registration_folder}")
                logging.debug(f"Deleted registration folder for user '{selected_user}'")
            if os.path.exists(verification_folder):
                os.system(f"rm -rf {verification_folder}")
                logging.debug(f"Deleted verification folder for user '{selected_user}'")

            logging.info(f"User '{selected_user}' deleted successfully")
            messagebox.showinfo("Success", f"User '{selected_user}' deleted successfully.")
            self.load_users()

    def update_user(self):
        selected_user = self.user_listbox.get(tk.ACTIVE)

        if not selected_user:
            logging.error("Update failed: No user selected")
            messagebox.showerror("Error", "Please select a user to update.")
            return

        new_keyword = simpledialog.askstring("Update User", f"Enter a new keyword for user '{selected_user}':")
        if not new_keyword:
            logging.error("Update failed: No new keyword provided")
            messagebox.showerror("Input Error", "Please enter a new keyword.")
            return

        try:
            logging.info(f"Updating user '{selected_user}' with new keyword '{new_keyword}'")
            self._register_user_logic(selected_user, new_keyword)
            logging.info(f"User '{selected_user}' updated successfully with new keyword '{new_keyword}'")
            messagebox.showinfo("Success", f"User '{selected_user}' updated successfully with new keyword '{new_keyword}'.")
            self.create_user_management_screen()
        except Exception as e:
            logging.error(f"Error during update for user '{selected_user}': {str(e)}")
            messagebox.showerror("Error", str(e))

# Create the Tkinter window
main_window = tk.Tk()
app = VoiceAuthApp(main_window)
main_window.mainloop()
