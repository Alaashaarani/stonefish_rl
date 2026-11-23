"""A GUI panel that can also own a standalone window."""

import tkinter as tk
from tkinter import ttk


class GuiPanel(ttk.Frame):
    def __init__(self, master=None, *, title, geometry):
        self._owns_window = master is None
        self.window = tk.Tk() if master is None else master.winfo_toplevel()
        super().__init__(self.window if master is None else master)
        if self._owns_window:
            self.window.title(title)
            self.window.geometry(geometry)
            self.pack(fill=tk.BOTH, expand=True)

    def geometry(self, value):
        return self.window.geometry(value)

    def protocol(self, *args):
        return self.window.protocol(*args)

    def destroy(self):
        if self._owns_window:
            self._owns_window = False
            self.window.destroy()
        else:
            super().destroy()
