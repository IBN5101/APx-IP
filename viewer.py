import mujoco
from mujoco import viewer

c = 2

if (c == 1):
    # Humanoid
    model = mujoco.MjModel.from_xml_path('humanoid.xml')
elif (c == 2):
    # Blobby
    model = mujoco.MjModel.from_xml_path('blobby.xml')
elif (c == 3):
    # Other
    model = mujoco.MjModel.from_xml_path('slider_crank.xml')

viewer.launch(model)