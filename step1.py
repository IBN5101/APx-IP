import mujoco

import sys
import cv2

model = mujoco.MjModel.from_xml_path('blobby.xml')
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

def contact_with_floor(data):
    for i in range(data.ncon):
        contact = data.contact[i]
        floor = None
        if data.geom(contact.geom1).name.startswith("floor"):
            floor = contact.geom1
        elif data.geom(contact.geom2).name.startswith("floor"):
            floor = contact.geom2
        shin = None
        if data.geom(contact.geom1).name.startswith("shin"):
            shin = contact.geom1
        elif data.geom(contact.geom2).name.startswith("shin"):
            shin = contact.geom2

        if (floor is not None) and (shin is not None):
            return True

    return False

# Settings (?)
scene_option = mujoco.MjvOption()

# Renderer
renderer = mujoco.Renderer(model, 240, 320)

print(model.body(1).pos)

# Simulation #1
duration = 1
mujoco.mj_resetData(model, data)
while data.time < duration:
    mujoco.mj_step(model, data)
print("Simulation completed")
# print(contact_with_floor(data))

# print(data.contact.geom1)
# print(data.contact.geom2)

# print(data.geom_xpos[1])
# data.geom_xpos[1][2] = 2
# print(data.geom(1))

# Simulation #2
duration = 10
framerate = 60

frames = []
mujoco.mj_resetData(model, data)
while data.time < duration:
    mujoco.mj_step(model, data)
    if (contact_with_floor(data)):
        print("HERE")
        data.geom_xpos[0][2] = data.geom_xpos[0][2] - 1
    if (len(frames) < data.time * framerate):
        renderer.update_scene(data, 0)
        pixels = renderer.render().copy()
        frames.append(pixels)
print("Simulation completed")
print(model.body(1).pos)

for i in range(len(frames)):
    cv2.imshow('Video', frames[i])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

