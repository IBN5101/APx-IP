import mujoco

import sys
import cv2

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
     rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>

  <worldbody>
    <geom size=".2 .2 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.1 .07" xyaxes="1 0 0 0 1 2"/>
    <body name="top" pos="0 0 .02">
      <freejoint/>
      <geom name="ball" type="sphere" size=".02" />
      <geom name="stem" type="cylinder" pos="0 0 .02" size="0.004 .008"/>
      <geom name="ballast" type="box" size=".023 .023 0.005"  pos="0 0 -.015"
       contype="0" conaffinity="0" group="3"/>
    </body>
  </worldbody>

  <keyframe>
    <key name="spinning" qpos="0 0 0.02 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
</mujoco>
"""

# Settings (?)
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Model & Data
model = mujoco.MjModel.from_xml_string(tippe_top)
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

# Simulation
duration = 10
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