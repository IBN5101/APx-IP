import mujoco

import sys
import cv2

# Model & Data
chaotic_pendulum = """
<mujoco>
  <option timestep=".001">
    <flag energy="enable" contact="disable"/>
  </option>

  <default>
    <joint type="hinge" axis="0 -1 0"/>
    <geom type="capsule" size=".02"/>
  </default>

  <worldbody>
    <light pos="0 -.4 1"/>
    <camera name="fixed" pos="0 -1 0" xyaxes="1 0 0 0 0 1"/>
    <body name="0" pos="0 0 .2">
      <joint name="root"/>
      <geom fromto="-.2 0 0 .2 0 0" rgba="1 1 0 1"/>
      <geom fromto="0 0 0 0 0 -.25" rgba="1 1 0 1"/>
      <body name="1" pos="-.2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="1 0 0 1"/>
      </body>
      <body name="2" pos=".2 0 0">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 1 0 1"/>
      </body>
      <body name="3" pos="0 0 -.25">
        <joint/>
        <geom fromto="0 0 0 0 0 -.2" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(chaotic_pendulum)
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)

# Settings (?)
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# Renderer
renderer = mujoco.Renderer(model, 240, 320)

# Simulation
duration = 10
framerate = 60

frames = []
mujoco.mj_resetData(model, data)
data.joint('root').qvel = 10
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