import mujoco

import sys
import cv2

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Settings (?)
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Model & Data
model = mujoco.MjModel.from_xml_string(xml)
# model = mujoco.MjModel.from_xml_path('humanoid.xml')
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)
# Renderer
renderer = mujoco.Renderer(model)

# Step
# mujoco.mj_forward(model, data)
# renderer.update_scene(data)

# cv2.imshow("Model", renderer.render())
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Physics
print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, 10)
print('flipped gravity', model.opt.gravity)

# Simulation
duration = 3.8
framerate = 60

frames = []
mujoco.mj_resetData(model, data)
while data.time < duration:
    mujoco.mj_step(model, data)
    if (len(frames) < data.time * framerate):
        renderer.update_scene(data)
        pixels = renderer.render().copy()
        frames.append(pixels)

for i in range(len(frames)):
    cv2.imshow('Video', frames[i])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()


print("success")