import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os.path 
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import os 

model_path = 'models/RRDB_ESRGAN_x4.pth' 
device = torch.device('cuda') 

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

def show_page(page_num):
    notebook.select(page_num)

def main_page():
    def compress_image():
        global original_image, save_path
        filepath = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                              filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")))
        if not filepath:
            return
        
        original_image = Image.open(filepath)
        
        original_image.thumbnail((300, 300))
        original_image_tk = ImageTk.PhotoImage(original_image)
        original_label.config(image=original_image_tk)
        original_label.image = original_image_tk
        
        select_button.filename = os.path.basename(filepath) 
        select_button.config(text=f"Select Image\n{select_button.filename}")
        
        confirm_button.config(state=tk.DISABLED)
        restart_button.config(state=tk.NORMAL) 
        choose_path_button.config(state=tk.NORMAL) 

    def choose_save_path():
        global save_path
        save_path = filedialog.askdirectory(initialdir="/", title="Select Directory to Save Compressed Image")
        if save_path:
            choose_path_button.config(text=f"Choose Save Path\n{save_path}")  
            confirm_button.config(state=tk.NORMAL)

    def confirm_compress():
        global original_image, save_path
        if original_image:
            if save_path:
                compressed_image = original_image.copy()
                compressed_image.save(f"{save_path}/compressed_image.jpg", optimize=True, quality=50) 
                status_label.config(text=f"Image compressed and saved as '{save_path}/compressed_image.jpg'")

    def restart():
        global original_image, save_path
        original_image = None
        save_path = None
        select_button.config(text="Select Image")
        original_label.config(image="")
        choose_path_button.config(text="Choose Save Path", state=tk.DISABLED)
        status_label.config(text="")
        restart_button.config(state=tk.DISABLED)
        confirm_button.config(state=tk.DISABLED) 

    main_frame = ttk.Frame(notebook)

    select_button = ttk.Button(main_frame, text="Select Image", command=compress_image)
    select_button.pack(padx=10, pady=5)

    choose_path_button = ttk.Button(main_frame, text="Choose Save Path", command=choose_save_path, state=tk.DISABLED)
    choose_path_button.pack(padx=10, pady=5)

    confirm_button = ttk.Button(main_frame, text="Confirm Compression", command=confirm_compress, state=tk.DISABLED)
    confirm_button.pack(padx=10, pady=5)

    original_label = ttk.Label(main_frame)
    original_label.pack(padx=10, pady=5)

    status_label = ttk.Label(main_frame, text="")
    status_label.pack(padx=10, pady=5)

    restart_button = ttk.Button(main_frame, text="Restart", command=restart, state=tk.DISABLED)
    restart_button.pack(padx=10, pady=5)

    return main_frame

def secondary_page():
    def select_image():
        global original_image, save_path
        filepath = filedialog.askopenfilename(initialdir="/", title="Select Image File",
                                            filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*")))
        if not filepath:
            return
        
        original_image = Image.open(filepath)
        
        original_image.thumbnail((300, 300))
        original_image_tk = ImageTk.PhotoImage(original_image)
        
        selected_image_label_second.config(image=original_image_tk)
        selected_image_label_second.image = original_image_tk
        
        select_button_second.filename = os.path.basename(filepath) 
        select_button_second.config(text=f"Select Image\n{select_button_second.filename}")
        
        save_path = None
        save_path_button_second.config(text="Choose Save Path", state=tk.NORMAL)  
        compress_button_second.config(text="Confirm Compression", state=tk.DISABLED)  
        status_label_second.config(text="")  
        restart_button_second.config(state=tk.NORMAL) 

    def choose_save_path():
        global save_path
        save_path = filedialog.askdirectory(initialdir="/", title="Select Directory to Save Compressed Image")
        if save_path:
            save_path_button_second.config(text=f"Choose Save Path\n{save_path}")  
            compress_button_second.config(state=tk.NORMAL) 

    def compress_image():
        global original_image, save_path
        if original_image:
            if save_path:

                img_np = np.array(original_image)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                img_np = img_np * 1.0 / 255
                img_np = torch.from_numpy(np.transpose(img_np[:, :, [2, 1, 0]], (2, 0, 1))).float()
                img_LR = img_np.unsqueeze(0)
                img_LR = img_LR.to(device)

                with torch.no_grad():
                    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round()

                cv2.imwrite(f'{save_path}/compressed_image.png', output)
                status_label_second.config(text=f"Image compressed and saved as '{save_path}/compressed_image.png'")
            else:
                status_label_second.config(text="Please choose a save path")
        else:
            status_label_second.config(text="Please select an image first")

    def restart():
        global original_image, save_path
        original_image = None
        save_path = None
        selected_image_label_second.config(image="")
        select_button_second.config(text="Select Image")
        save_path_button_second.config(text="Choose Save Path", state=tk.DISABLED)
        compress_button_second.config(text="Confirm Compression", state=tk.DISABLED)
        status_label_second.config(text="")
        restart_button_second.config(state=tk.DISABLED)

    secondary_frame = ttk.Frame(root)

    select_button_second = ttk.Button(secondary_frame, text="Select Image", command=select_image)
    select_button_second.pack(padx=10, pady=5)

    selected_image_label_second = ttk.Label(secondary_frame)
    selected_image_label_second.pack(padx=10, pady=5)

    save_path_button_second = ttk.Button(secondary_frame, text="Choose Save Path", command=choose_save_path, state=tk.DISABLED)
    save_path_button_second.pack(padx=10, pady=5)

    compress_button_second = ttk.Button(secondary_frame, text="Confirm Compression", command=compress_image, state=tk.DISABLED)
    compress_button_second.pack(padx=10, pady=5)

    restart_button_second = ttk.Button(secondary_frame, text="Restart", command=restart, state=tk.DISABLED)
    restart_button_second.pack(padx=10, pady=5)

    status_label_second = ttk.Label(secondary_frame, text="")
    status_label_second.pack(padx=10, pady=5)
    return secondary_frame

def show_before_after_images(original_image, compressed_image):
    original_image.thumbnail((300, 300))
    compressed_image.thumbnail((300, 300))
    original_image_tk = ImageTk.PhotoImage(original_image)
    compressed_image_tk = ImageTk.PhotoImage(compressed_image)

    before_after_frame = ttk.Frame(root)
    before_label = ttk.Label(before_after_frame, text="Before Compression")
    after_label = ttk.Label(before_after_frame, text="After Compression")
    before_label.grid(row=0, column=0, padx=10, pady=5)
    after_label.grid(row=0, column=1, padx=10, pady=5)

    original_label = ttk.Label(before_after_frame, image=original_image_tk)
    compressed_label = ttk.Label(before_after_frame, image=compressed_image_tk)
    original_label.image = original_image_tk
    compressed_label.image = compressed_image_tk
    original_label.grid(row=1, column=0, padx=10, pady=5)
    compressed_label.grid(row=1, column=1, padx=10, pady=5)

    return before_after_frame

root = tk.Tk()
root.title("SuperMod")
root.geometry("800x600")

notebook = ttk.Notebook(root)

main_page = main_page()
secondary_page = secondary_page()

notebook.add(main_page, text='Main Page')
notebook.add(secondary_page, text='Secondary Page')

notebook.pack(padx=10, pady=10, fill='both', expand=True)

original_image = None
save_path = None

show_page(0)

root.mainloop()
