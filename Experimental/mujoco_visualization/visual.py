import mujoco

import sys
import cv2

model = mujoco.MjModel.from_xml_path('blobby.xml')
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

# Settings (?)
scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Renderer
renderer = mujoco.Renderer(model, 240, 320)

# Simulation
duration = 10
framerate = 60

frames = []
mujoco.mj_resetData(model, data)
while data.time < duration:
    mujoco.mj_step(model, data)
    if (len(frames) < data.time * framerate):
        renderer.update_scene(data, 0)
        pixels = renderer.render().copy()
        frames.append(pixels)
print("Simulation completed")

for i in range(len(frames)):
    cv2.imshow('Video', frames[i])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


print("success")