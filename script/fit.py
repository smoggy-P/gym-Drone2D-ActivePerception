#!/usr/bin/python3.9

import pulp
import numpy as np
import pickle
import tkinter as tk
from PIL import Image, ImageTk
import gym
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utils import Params
import main
import pygame
import os
from tkinter import font, Label, Entry, Button
from numpy.linalg import norm

import sys
sys.path.append('/home/smoggy/thesis/gym-Drone2D-ActivePerception/')

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['SDL_AUDIODRIVER'] = 'dsp'

def calculate_surv(env, params):
    position_step = 60
    T = 12
    x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)
    survive_times = np.ones((len(x_range), len(y_range))) * T
    _, _, done, info = env.step(0)
    for t in np.arange(0, T, 0.1):
        for x in x_range:
            for y in y_range:
                drone_pos = np.array([x, y])
                for agent in env.agents:
                    if norm(agent.position - drone_pos) < agent.radius + env.drone.radius:
                        survive_times[x_range.index(x), y_range.index(y)] = min(t, survive_times[x_range.index(x), y_range.index(y)])
        _, _, done, info = env.step(0)
    survive_times = survive_times - 0.1
    survive_times[survive_times < 0] = 0
    return 10-10*(np.mean(survive_times)-1.73)/(10.64-1.73)

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
        global photo
        params = Params(agent_number=agent_num.varValue,
                        agent_radius=agent_size.varValue*10,
                        agent_max_speed=agent_vel.varValue*10,
                        planner='NoMove')
        env = gym.make('gym-2d-perception-v2', params=params)
        env.reset()
        env.step(0)
        env.render()
        rgb_array = pygame.surfarray.array3d(env.screen)

        new_img = Image.fromarray(np.transpose(rgb_array, (1, 0, 2)))
        photo = ImageTk.PhotoImage(new_img)
        canvas.itemconfig(image_container,image=photo)

        result_var.set(f"Obstacle Number={int(agent_num.varValue)}\n Obstacle Size={agent_size.varValue:.2f}\n Obstacle Velocity={agent_vel.varValue:.2f}\n Actual Survivability Metric={calculate_surv(env, params):.2f}")
    else:
        result_var.set("Optimization failed. Try different inputs or check model.")
    

filename = './script/fit_model.sav'
reg = pickle.load(open(filename, 'rb'))

# Initialize main window
root = tk.Tk()
root.title("GUI with Image")
root.geometry('1200x650')  # Adjust as per your image size and other elements

# Set the default font
default_font = font.nametofont("TkDefaultFont")
default_font.configure(family='"Arial Bold"', size=16)

# Style for the entry widgets
entry_style = {'relief': tk.SUNKEN, 'borderwidth': 3}

# Variables
y_input = tk.StringVar()
agent_num_input = tk.StringVar()
agent_size_input = tk.StringVar()
agent_vel_input = tk.StringVar()
result_var = tk.StringVar()

# Load an image from a directory
image_path = '/home/smoggy/thesis/paper/thesis/figures/dense1.png'
img = Image.open(image_path)
photo = ImageTk.PhotoImage(img)

# Heading
heading_label = tk.Label(root, text="Obstacle Environment Generator", font=('Arial', 24), pady=20)
heading_label.grid(row=0, column=0, columnspan=3)

# Canvas for Image
canvas = tk.Canvas(root, width=img.width, height=img.height, bg='white', bd=0, highlightthickness=0)
canvas.grid(row=1, column=2, rowspan=6, padx=0, pady=0)
image_container = canvas.create_image(img.width//2, img.height//2, image=photo) 

# Settings Frame
settings_frame = tk.Frame(root, bg='#f2f2f2', padx=10, pady=10)
settings_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=20)

# Layout for settings inside frame
Label(settings_frame, text="Desired Survivability Metric(0-10):", font=default_font, bg='#f2f2f2').grid(row=0, column=0, sticky='e', padx=5, pady=5)
Entry(settings_frame, textvariable=y_input, font=default_font, **entry_style).grid(row=0, column=1, padx=5, pady=5)

Label(settings_frame, text="Obstacle Number (optional):", font=default_font, bg='#f2f2f2').grid(row=1, column=0, sticky='e', padx=5, pady=5)
Entry(settings_frame, textvariable=agent_num_input, font=default_font, **entry_style).grid(row=1, column=1, padx=5, pady=5)

Label(settings_frame, text="Obstacle Size (optional):", font=default_font, bg='#f2f2f2').grid(row=2, column=0, sticky='e', padx=5, pady=5)
Entry(settings_frame, textvariable=agent_size_input, font=default_font, **entry_style).grid(row=2, column=1, padx=5, pady=5)

Label(settings_frame, text="Obstacle Velocity (optional):", font=default_font, bg='#f2f2f2').grid(row=3, column=0, sticky='e', padx=5, pady=5)
Entry(settings_frame, textvariable=agent_vel_input, font=default_font, **entry_style).grid(row=3, column=1, padx=5, pady=5)

Button(root, text="Generate", command=predict, font=default_font, relief=tk.RAISED, borderwidth=3, bg="#4CAF50", fg="white").grid(row=2, column=0, columnspan=2, padx=10, pady=10)
# Frame to encapsulate the results for better aesthetics
result_frame = tk.Frame(root, bg='#e6e6e6', padx=10, pady=10, relief=tk.GROOVE, borderwidth=2)
result_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

# Adjust column weights so that the content is centered in the frame
result_frame.grid_columnconfigure(0, weight=1)

# Display the result_var within the new frame
Label(result_frame, textvariable=result_var, font=default_font, bg='#e6e6e6').grid(row=0, column=0)

# Main loop to run the GUI
root.mainloop()
