# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:54:02 2021

@author: sp3660
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
#--------------------------------------------------------------
fig = Figure(figsize=(12, 8), facecolor='white')
#--------------------------------------------------------------
axis = fig.add_subplot(111) # 1 row, 1 column, only graph #<--uncomment
# axis = fig.add_subplot(211) # 2 rows, 1 column, Top graph
#--------------------------------------------------------------

xValues = [1,2,3,4]
yValues = [5,7,6,8]
axis.plot(xValues, yValues)
axis.set_xlabel('Horizontal Label')
axis.set_ylabel('Vertical Label')
# axis.grid() # default line style
axis.grid(linestyle='-') # solid grid lines

#--------------------------------------------------------------
def _destroyWindow():
  root.quit()
  root.destroy()
#--------------------------------------------------------------
root = tk.Tk()
root.protocol('WM_DELETE_WINDOW', _destroyWindow)
#--------------------------------------------------------------

canvas = FigureCanvasTkAgg(fig, master=root)
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#--------------------------------------------------------------
root.mainloop()