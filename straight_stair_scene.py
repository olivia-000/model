import asyncio
import numpy as np
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import FixedCuboid

def box(world, name, p, s, c):
    obj = FixedCuboid(prim_path=f"/World/{name}", name=name,
        position=np.array(p, dtype=np.float32),
        scale=np.array(s, dtype=np.float32),
        color=np.array(c, dtype=np.float32))
    world.scene.add(obj)

async def main():
    if World.instance():
        World.instance().clear_instance()
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()

    FLOOR_T=0.12
    STEP_H=0.10
    STEP_D=0.30
    WIDTH=1.50
    N=10
    TOP_Z=N*STEP_H
    C_F=[0.42,0.42,0.42]
    C_S=[0.72,0.72,0.72]
    C_2=[0.38,0.52,0.68]

    # 1F
    box(world,"Floor_1",[0,0,-FLOOR_T/2],[16,10,FLOOR_T],C_F)

    # Straight stair: +X
    x0=-4.0
    y0=0.0
    for i in range(N):
        top=STEP_H*(i+1)
        x=x0+STEP_D/2+i*STEP_D
        box(world,f"StraightStair_{i:02d}",[x,y0,top/2],
            [STEP_D,WIDTH,top],C_S)

    exit_x=x0+N*STEP_D

    # Upper floor
    L=7.0
    box(world,"Floor_2",[exit_x+L/2,y0,TOP_Z-FLOOR_T/2],
        [L,7.0,FLOOR_T],C_2)

    await world.reset_async()
    print(f"Straight stair: steps={N}, top={TOP_Z:.2f} m")

asyncio.ensure_future(main())
