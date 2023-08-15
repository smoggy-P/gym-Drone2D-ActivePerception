#!/usr/bin/python3.9

import pulp
import numpy as np
import pickle
import tkinter as tk
from PIL import Image, ImageTk
# import main
import gym
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utils import Params
import main
import pygame
import os
from tkinter import font, Label, Entry, Button

import sys
sys.path.append('/home/smoggy/thesis/gym-Drone2D-ActivePerception/')

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['SDL_AUDIODRIVER'] = 'dsp'



filename = './script/fit_model.sav'
reg = pickle.load(open(filename, 'rb'))
def update_img(agent_num, agent_size, agent_vel):
    global photo
    params = Params(agent_number=agent_num,
                    agent_radius=agent_size*10,
                    agent_max_speed=agent_vel*10,
                    planner='NoMove')
    env = gym.make('gym-2d-perception-v2', params=params)
    env.reset()
    env.step(0)
    env.render()
    rgb_array = pygame.surfarray.array3d(env.screen)
    new_img = Image.fromarray(np.transpose(rgb_array, (1, 0, 2)))
    photo = ImageTk.PhotoImage(new_img)
    img_label.config(image=photo)
    img_label.update()


def predict():
    try:
        y_value = float(y_input.get())
    except ValueError:
        result_var.set("Invalid y input. Please enter a number.")
        return

    model = pulp.LpProblem("Prediction", pulp.LpMinimize)

    # Variables with bounds
    agent_num = pulp.LpVariable('agent_num', lowBound=10, upBound=30, cat='Integer')
    agent_size = pulp.LpVariable('agent_size', lowBound=0.5, upBound=1.5)
    agent_vel = pulp.LpVariable('agent_vel', lowBound=2, upBound=6)

    # If input is provided, set variable to that value
    if agent_num_input.get():
        model += agent_num == float(agent_num_input.get())
    if agent_size_input.get():
        model += agent_size == float(agent_size_input.get())
    if agent_vel_input.get():
        model += agent_vel == float(agent_vel_input.get())
    z = pulp.LpVariable('z', lowBound=0)  # Represents the absolute difference
    # Add the constraint that z is at least the positive difference
    model += z >= reg.predict([[agent_num, agent_size, agent_vel]])[0] - y_value
    # Add the constraint that z is at least the negative difference
    model += z >= y_value - reg.predict([[agent_num, agent_size, agent_vel]])[0]
    # Minimize z
    model += z
    # Solve the problem
    model.solve()
    if pulp.LpStatus[model.status] == 'Optimal':
        result_var.set(f"agent_num={int(agent_num.varValue)}, agent_size={agent_size.varValue:.2f}, agent_vel={agent_vel.varValue:.2f}")
    else:
        result_var.set("Optimization failed. Try different inputs or check model.")
    
    update_img(agent_num.varValue, agent_size.varValue, agent_vel.varValue)

# Create main window
root = tk.Tk()
root.title("GUI with Image")

# Set the default font
default_font = font.nametofont("TkDefaultFont")
default_font.configure(family='Arial', size=20)
font.families()
# Variables
y_input = tk.StringVar()
agent_num_input = tk.StringVar()
agent_size_input = tk.StringVar()
agent_vel_input = tk.StringVar()
result_var = tk.StringVar()

# Load an image from a directory
image_path = '/home/smoggy/thesis/paper/thesis/figures/dense1.png'  # Replace with your image's path
img = Image.open(image_path)
photo = ImageTk.PhotoImage(img)

# Add a label with the image
img_label = tk.Label(root, image=photo)
img_label.grid(row=0, column=2, rowspan=6, padx=10, pady=10)  # Adjust the position as required

# Layout for the rest of your GUI elements
Label(root, text="Enter y value:", font=default_font).grid(row=0, column=0, padx=5, pady=5, sticky='e')
Entry(root, textvariable=y_input, font=default_font).grid(row=0, column=1, padx=5, pady=5)

Label(root, text="agent_num (optional):", font=default_font).grid(row=1, column=0, padx=5, pady=5, sticky='e')
Entry(root, textvariable=agent_num_input, font=default_font).grid(row=1, column=1, padx=5, pady=5)

Label(root, text="agent_size (optional):", font=default_font).grid(row=2, column=0, padx=5, pady=5, sticky='e')
Entry(root, textvariable=agent_size_input, font=default_font).grid(row=2, column=1, padx=5, pady=5)

Label(root, text="agent_vel (optional):", font=default_font).grid(row=3, column=0, padx=5, pady=5, sticky='e')
Entry(root, textvariable=agent_vel_input, font=default_font).grid(row=3, column=1, padx=5, pady=5)

Button(root, text="Predict", command=predict, font=default_font).grid(row=4, column=0, columnspan=2, padx=5, pady=5)
Label(root, textvariable=result_var, font=default_font).grid(row=5, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
