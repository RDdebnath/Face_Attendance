import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

def start_attendance():
    try:
        # Adjust 'python' command to 'python3' if needed
        subprocess.Popen([sys.executable, os.path.join(os.getcwd(), 'attendance.py')])
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x200")

label = tk.Label(root, text="Smart Attendance System", font=("Helvetica", 16))
label.pack(pady=20)

start_btn = tk.Button(root, text="Start Attendance", command=start_attendance, width=20, height=2)
start_btn.pack(pady=10)

root.mainloop()
