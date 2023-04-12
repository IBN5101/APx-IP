import mujoco
from mujoco import viewer

c = 1

if (c == 1):
    # Humanoid
    model = mujoco.MjModel.from_xml_path('Experimental/MuJoCo/humanoid.xml')
elif (c == 2):
    # Blobby
    model = mujoco.MjModel.from_xml_path('blobby.xml')
elif (c == 3):
    # Other
    model = mujoco.MjModel.from_xml_path('Experimental/MuJoCo/ant.xml')

viewer.launch(model)