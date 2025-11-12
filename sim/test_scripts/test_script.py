# from isaacsim import SimulationApp

# import carb

# # Simple example showing how to start and stop the helper
# simulation_app = SimulationApp({"headless": True})

# ### Perform any omniverse imports here after the helper loads ###

# simulation_app.update()  # Render a single frame
# simulation_app.close()  # Cleanup application

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

DynamicCuboid(
   prim_path="/new_cube_8",
   name="cube_8",
   position=np.array([5.0, 0, 2.0]),
   scale=np.array([0.6, 0.5, 0.2]),
   size=1.0,
   color=np.array([0, 255, 0]),
)