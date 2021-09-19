# # -*- coding: utf-8 -*-
# """
# Created on Sun Jun 20 09:02:40 2021

# @author: sp3660
# """

# import tkinter as tk
# from tkinter import ttk
# from tkinter import scrolledtext
# from tkinter import Menu
# from tkinter import messagebox as msg
# from time import sleep 
# import matplotlib.pyplot as plt
# from pylab import show


# class ToolTip(object):
#     def __init__(self, widget, tip_text=None):
#         self.widget = widget
#         self.tip_text = tip_text
#         widget.bind('<Enter>', self.mouse_enter)
#         widget.bind('<Leave>', self.mouse_leave)
        
#     def mouse_enter(self, _event):
#         self.show_tooltip()
        
#     def mouse_leave(self, _event):
#         self.hide_tooltip()
        
#     def show_tooltip(self):
#          if self.tip_text:
#              x_left = self.widget.winfo_rootx()
#              y_top = self.widget.winfo_rooty() - 18
             
#              self.tip_window = tk.Toplevel(self.widget)
#              self.tip_window.overrideredirect(True)
#              self.tip_window.geometry("+%d+%d" % (x_left, y_top))
#              label = tk.Label(self.tip_window, text=self.tip_text,justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID,borderwidth=1, font=("tahoma", "8", "normal"))
#              label.pack(ipadx=1)
             
#     def hide_tooltip(self):
#          if self.tip_text:
#              self.tip_window.destroy()


# GLOBAL_CONST = 42
# # ...
# print(GLOBAL_CONST)

# win = tk.Tk()
# win.title("Python GUI")
# # win.iconbitmap(r'C:\Users\sp3660\Downloads\images.ico')

# tabControl = ttk.Notebook(win)
# tab1=ttk.Frame(tabControl)
# tab2=ttk.Frame(tabControl)
# tabControl.add(tab1, text='Tab 1')
# tabControl.add(tab2, text='Tab 2')
# tab3 = ttk.Frame(tabControl)
# tabControl.add(tab3, text='Tab 3') 
# tabControl.pack(expand=1, fill="both")


# doubleData = tk.DoubleVar()
# print(doubleData.get())
# doubleData.set(2.4)
# print(type(doubleData))
# add_doubles = 1.222222222222222222222222 + doubleData.get()
# print(add_doubles)
# print(type(add_doubles))
# strData = tk.StringVar()
# strData.set('Hello StringVar')
# varData = strData.get()
# print(varData)
# print(tk.IntVar())
# print(tk.DoubleVar())
# print(tk.BooleanVar())
# intData = tk.IntVar()
# print(intData)
# print(intData.get())

# # tab1.resizable(False, False)

# ttk.Label(tab1, text="Enter a name").grid(column=0, row=1)

# second_label=ttk.Label(tab1, text="Second Label")
# second_label.grid(column=0, row=0)

# name = tk.StringVar()
# name_entered = ttk.Entry(tab1, width=12, textvariable=name)
# name_entered.grid(column=0, row=2)

# ttk.Label(tab1, text="Choose a number:").grid(column=1, row=1)
# number = tk.StringVar()
# number_chosen = ttk.Combobox(tab1, width=12, textvariable=number,state='readonly')
# number_chosen['value'] = (1, 2, 4, 42, 100)
# number_chosen.grid(column=1, row=2)
# number_chosen.current(2)

# chVarDis = tk.IntVar()
# chVarUn = tk.IntVar()
# chVarEn = tk.IntVar()
# check1=tk.Checkbutton(tab1, text="Disabled", variable=chVarDis, state='disabled')
# check2=tk.Checkbutton(tab1, text="UnChecked", variable=chVarUn)
# check3=tk.Checkbutton(tab1, text="Enabled", variable=chVarEn)
# check1.select()
# check2.deselect()
# check3.select()
# check1.grid(column=0, row=4, sticky=tk.W)
# check2.grid(column=1, row=4, sticky=tk.W)
# check3.grid(column=2, row=4, sticky=tk.W)

# colors = ["Blue", "Gold", "Red"]
# radVar=tk.IntVar()
# radVar.set(99)

# scrol_w=30
# scrol_h=3
# scr = scrolledtext.ScrolledText(tab1, width=scrol_w, height=scrol_h,
# wrap=tk.WORD)
# scr.grid(column=0, row=6, columnspan=3)

# buttons_frame=ttk.LabelFrame(tab1, text='Labels in a Frame') 
# buttons_frame.grid(column=0, row=10, padx=20, pady=40 )
# buttons_frame.grid(column=1, row=10)

# ttk.Label(buttons_frame, text="Label1 -- jkdshflsdkjafhsldjfhsdalfhkjdsf").grid(column=0, row=0, sticky=tk.W)
# ttk.Label(buttons_frame, text="Label2").grid(column=0, row=1, sticky=tk.W)
# ttk.Label(buttons_frame, text="Label3").grid(column=0, row=2, sticky=tk.W)

# for child in buttons_frame.winfo_children():
#  child.grid_configure(padx=8, pady=4)
# for child in tab1.winfo_children():
#  child.grid_configure(padx=8, pady=4)

# def radCall():
#     radSel=radVar.get()
#     if radSel == 0: tab1.configure(background=colors[0])
#     elif radSel == 1: tab1.configure(background=colors[1])
#     elif radSel == 2: tab1.configure(background=colors[2])

# def click_me():
#     action.configure(text="** I have been Clicked! **")
#     second_label.configure(foreground='red')
#     second_label.configure(text='A Red Label')
    
# def click_me2():
#     action2.configure(text='Hello ' + name.get() + ' ' + number_chosen.get())
#     second_label.configure(text='Hello ' + name.get() + ' ' + number_chosen.get())
  
# action = ttk.Button(tab1, text="Click Me!", command=click_me)
# action.grid(column=1, row=0)

# action2 = ttk.Button(tab1, text="Click Me!", command=click_me2)
# action2.grid(column=2, row=0)
# # action2.configure(state='disabled')

# for col in range(3):
#  curRad = tk.Radiobutton(tab1, text=colors[col], variable=radVar, value=col, command=radCall)
#  curRad.grid(column=col, row=5, sticky=tk.W)
 

 
 
 
# progress_bar = ttk.Progressbar(tab2, orient='horizontal', length=286, mode='determinate')
# progress_bar.grid(column=0, row=3, pady=2)
 
# def start_progressbar():
#     progress_bar.start()
# def stop_progressbar():
#     progress_bar.stop()
# def progressbar_stop_after(wait_ms=1000):
#     win.after(wait_ms, progress_bar.stop)
 
# def run_progressbar():
#   progress_bar["maximum"] = 100
#   for i in range(101):
#       sleep(0.05)
#       progress_bar["value"] = i # increment progressbar
#       progress_bar.update() # have to call update() in loop
#       progress_bar["value"] = 0 
     
 
# # LabelFrame using tab1 as the parent
# mighty = ttk.LabelFrame(tab2, text=' Mighty Python ')
# mighty.grid(column=0, row=0, padx=8, pady=4)
# # Label using mighty as the parent
# a_label = ttk.Label(mighty, text="Enter a name:")
# a_label.grid(column=0, row=0, sticky='W') 
 
# mighty2=ttk.LabelFrame(tab2, text=' Mighty2 Python ')
# mighty2.grid(column=0, row=7, padx=8, pady=4)

# # NEW CODE
# # Create a container to hold buttons
# buttons_frame2 = ttk.LabelFrame(mighty2, text=' ProgressBar ')
# buttons_frame2.grid(column=0, row=2, sticky='W', columnspan=2) 
# # Add Buttons for Progressbar commands
# ttk.Button(buttons_frame2, text=" Run Progressbar ",command=run_progressbar).grid(column=0, row=0, sticky='W')
# ttk.Button(buttons_frame2, text=" Start Progressbar ",command=start_progressbar).grid(column=0, row=1, sticky='W')
# ttk.Button(buttons_frame2, text=" Stop immediately ",command=stop_progressbar).grid(column=0, row=2, sticky='W')
# ttk.Button(buttons_frame2, text=" Stop after second ",command=progressbar_stop_after).grid(column=0, row=3, sticky='W')




# for child in buttons_frame2.winfo_children():
#  child.grid_configure(padx=2, pady=2)
# for child in mighty2.winfo_children():
#  child.grid_configure(padx=8, pady=2)
 
# def _spin():
#  value = spin.get()
#  print(value)
#  scr.insert(tk.INSERT, value + '\n') # <-- add a newline
# def _spin2():
#  value = spin2.get()
#  print(value)
#  scr.insert(tk.INSERT, value + '\n')
 
 
# # spin = ttk.Spinbox(mighty, from_=0, to=10, style='Test', command=_spin)
# # s=ttk.Style(spin) 
# # s.configure('Test', width=5, bd=8)

# spin = tk.Spinbox(mighty, from_=0, to=10, width=5, bd=8, command=_spin)
# spin.grid(column=0, row=2) 
# spin2 = tk.Spinbox(mighty, width=5, bd=5, command=_spin, relief=tk.RIDGE)
# spin2['values'] = (1, 2, 4, 42, 100)
# spin2.grid(column=0, row=3)
  
# ToolTip(number_chosen, 'This is a Spin control')
# ToolTip(scr, 'This is a ScrolledText widget') # <-- add this code

 
 
 
# tab3_frame = tk.Frame(tab3, bg='blue')
# tab3_frame.pack()
# for orange_color in range(2):
#  canvas = tk.Canvas(tab3_frame, width=150, height=80, highlightthickness=0, bg='orange')
#  canvas.grid(row=orange_color, column=orange_color) 
 
 
 
 
# strData = spin.get()
# print("Spinbox value: " + strData)
# def usingGlobal():
#     print(GLOBAL_CONST)
# # call the function
# usingGlobal()


 
 
 
# def _msgBox():
#  # msg.showinfo('Python Message Info Box', 'A Python GUI created using tkinter:\nThe year is 2019.')
#  # msg.showwarning('Python Message Warning Box', 'A Python GUI created using tkinter:' '\nWarning: There might be a bug in this code.')
#  # msg.showerror('Python Message Error Box', 'A Python GUI created using tkinter:''\nError: Houston ~ we DO have a serious PROBLEM!')
#  answer = msg.askyesnocancel("Python Message Multi Choice Box", "Are you sure you really wish to do this?")

 
 
 
 
# def _quit():
#     tab1.quit()
#     tab1.destroy()
#     exit()

# menu_bar = Menu(win)
# win.config(menu=menu_bar)

# file_menu = Menu(menu_bar, tearoff=0) # create File menu
# menu_bar.add_cascade(label="File", menu=file_menu) # add File menu to menu bar and give it a label
# file_menu.add_command(label="New") # add File menu item
# file_menu.add_separator()
# file_menu.add_command(label="Exit", command=_quit) # add File menu item

# help_menu=Menu(menu_bar, tearoff=0) # create File menu
# menu_bar.add_cascade(label="Help", menu=help_menu) # add File menu to menu bar and give it a label
# help_menu.add_command(label="About", command=_msgBox) # add File menu item








# name_entered.focus()




# tab1.mainloop()
