import os
import open3d as o3d
import tkinter as tk
from tkinter import ttk, messagebox
"""

"""

# ---- CONFIG ----
folder_path = r"E:\programming\PYTHON\SMPL-x\data\outputs\meshes\obj"


# ---- OPEN3D MESH VIEWER ----
def visualize_mesh(path):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        messagebox.showerror("Error", "Cannot read OBJ file!")
        return

    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=os.path.basename(path))
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()


# ---- WHEN SELECTING FILE ----
def open_selected():
    selected = listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning")
        return

    file_name = listbox.get(selected[0])
    file_path = os.path.join(folder_path, file_name)
    visualize_mesh(file_path)


def on_double_click(event):
    open_selected()


# ------------------ UI ------------------

root = tk.Tk()
root.title("OBJ Viewer - Dark Theme")

# Window size + centering
w, h = 400, 500
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
root.geometry(f"{w}x{h}+{int((ws-w)/2)}+{int((hs-h)/2)}")

# ---- DARK THEME COLORS ----
BG = "#1e1e1e"
FG = "#e0e0e0"
BTN_BG = "#3a7ff6"
BTN_HOVER = "#5a95ff"
LIST_BG = "#2b2b2b"
LIST_FG = "#dcdcdc"
SCROLL_BG = "#555"
SCROLL_ACTIVE = "#777"

root.configure(bg=BG)


# ---- TITLE ----
title = tk.Label(
    root,
    text="Select OBJ File",
    bg=BG,
    fg="#4fa3ff",
    font=("Segoe UI", 16, "bold")
)
title.pack(pady=15)


# ---- LIST FRAME ----
frame = tk.Frame(root, bg=BG)
frame.pack(fill="both", expand=True, padx=20, pady=10)

# ---- LISTBOX ----
listbox = tk.Listbox(
    frame,
    bg=LIST_BG,
    fg=LIST_FG,
    font=("Segoe UI", 12),
    selectbackground="#4fa3ff",
    selectforeground="#000000",
    activestyle="none",
    borderwidth=0,
    highlightthickness=1,
    highlightcolor="#4fa3ff",
    height=18
)
listbox.pack(side="left", fill="both", expand=True)

# ---- SCROLLBAR ----
scroll = tk.Scrollbar(frame)
scroll.pack(side="right", fill="y")
listbox.config(yscrollcommand=scroll.set)
scroll.config(command=listbox.yview)


# ---- BUTTON ----
def on_enter(e):
    open_btn["bg"] = BTN_HOVER
def on_leave(e):
    open_btn["bg"] = BTN_BG

open_btn = tk.Button(
    root,
    text="",
    bg=BTN_BG,
    fg="white",
    font=("Segoe UI", 12, "bold"),
    command=open_selected,
    relief="flat",
    height=2
)
open_btn.pack(pady=10)
open_btn.bind("<Enter>", on_enter)
open_btn.bind("<Leave>", on_leave)


# ---- LOAD OBJ FILES ----
for f in os.listdir(folder_path):
    if f.lower().endswith(".obj"):
        listbox.insert("end", f)

# Double click to open
listbox.bind("<Double-Button-1>", on_double_click)


root.mainloop()
