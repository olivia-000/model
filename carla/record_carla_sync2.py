import carla
import os
import queue
import numpy as np
import pygame
import random
from PIL import Image

# ==================== Settings ====================
OUTPUT_SEMANTIC = os.path.expanduser("~/carla/datasets/recorded_Town03/semantic")
OUTPUT_RGB = os.path.expanduser("~/carla/datasets/recorded_Town03/rgb")
os.makedirs(OUTPUT_SEMANTIC, exist_ok=True)
os.makedirs(OUTPUT_RGB, exist_ok=True)

NUM_FRAMES = 1000

MAPS = ['Town03', 'Town01', 'Town02', 'Town04', 'Town05']

WEATHERS = [
    ('Sunny',   carla.WeatherParameters(cloudiness=10,  precipitation=0,   sun_altitude_angle=70)),
    ('Cloudy',  carla.WeatherParameters(cloudiness=60,  precipitation=0,   sun_altitude_angle=60)),
    ('Rain',    carla.WeatherParameters(cloudiness=80,  precipitation=60,  sun_altitude_angle=40)),
    ('Dusk',    carla.WeatherParameters(cloudiness=20,  precipitation=0,   sun_altitude_angle=10)),
    ('Night',   carla.WeatherParameters(cloudiness=10,  precipitation=0,   sun_altitude_angle=-30)),
]
# ==================================================

def safe_destroy(actor):
    """安全刪除 actor，忽略已被刪除的錯誤"""
    try:
        if actor and actor.is_alive:
            actor.destroy()
    except Exception:
        pass

def get_manual_control(keys):
    control = carla.VehicleControl()
    if keys[pygame.K_UP]:
        control.throttle = 0.8
        control.reverse = False
    elif keys[pygame.K_DOWN]:
        control.throttle = 0.8
        control.reverse = True
    else:
        control.throttle = 0.0
        control.brake = 0.0
    if keys[pygame.K_LEFT]:
        control.steer = -0.5
    elif keys[pygame.K_RIGHT]:
        control.steer = 0.5
    else:
        control.steer = 0.0
    return control


def draw_ui(display, frame_idx, total_frames, recording, current_map, current_weather, paused, manual):
    font_big = pygame.font.Font(pygame.font.get_default_font(), 24)
    font_small = pygame.font.Font(pygame.font.get_default_font(), 18)

    overlay = pygame.Surface((320, 220), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    display.blit(overlay, (10, 10))

    if paused:
        status = 'PAUSED'
        color = (255, 200, 0)
    elif recording:
        status = 'RECORDING'
        color = (255, 50, 50)
    else:
        status = 'STANDBY'
        color = (200, 200, 200)

    display.blit(font_big.render(status, True, color), (20, 15))
    mode = 'Mode: MANUAL' if manual else 'Mode: AUTOPILOT'
    mode_color = (255, 150, 0) if manual else (100, 200, 255)
    display.blit(font_small.render(mode, True, mode_color), (20, 50))
    display.blit(font_small.render(f'Frame: {frame_idx} / {total_frames}', True, (255, 255, 255)), (20, 75))
    display.blit(font_small.render(f'Map: {current_map}', True, (180, 255, 180)), (20, 100))
    display.blit(font_small.render(f'Weather: {current_weather}', True, (180, 220, 255)), (20, 125))
    display.blit(font_small.render('R=Record  SPACE=Pause  Q=Quit', True, (200, 200, 200)), (20, 150))
    display.blit(font_small.render('A=Auto/Manual  W=Weather  M=Map', True, (200, 200, 200)), (20, 175))

    if manual:
        ctrl_overlay = pygame.Surface((280, 60), pygame.SRCALPHA)
        ctrl_overlay.fill((0, 0, 0, 150))
        display.blit(ctrl_overlay, (10, 440))
        display.blit(font_small.render('UP=Forward  DOWN=Reverse', True, (255, 200, 100)), (20, 445))
        display.blit(font_small.render('LEFT/RIGHT=Steer', True, (255, 200, 100)), (20, 468))


def spawn_npcs(world, blueprint_library, n=30):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    npcs = []
    for sp in spawn_points[:n]:
        bp = random.choice(blueprint_library.filter('vehicle.*'))
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, 8000)
            npcs.append(npc)
    return npcs


def setup_cameras(world, blueprint_library, cam_transform, vehicle, rgb_queue, sem_queue):
    rgb_bp = blueprint_library.find('sensor.camera.rgb')
    rgb_bp.set_attribute('image_size_x', '1024')
    rgb_bp.set_attribute('image_size_y', '512')
    rgb_bp.set_attribute('fov', '90')

    sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', '1024')
    sem_bp.set_attribute('image_size_y', '512')

    rgb_cam = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
    sem_cam = world.spawn_actor(sem_bp, cam_transform, attach_to=vehicle)
    rgb_cam.listen(rgb_queue.put)
    sem_cam.listen(sem_queue.put)
    return rgb_cam, sem_cam


def main():
    pygame.init()
    display = pygame.display.set_mode((1024, 512))
    pygame.display.set_caption('CARLA Recorder')
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)

    recording = True
    paused = False
    manual = False
    frame_idx = 0
    weather_idx = 0
    map_idx = 0
    current_map = MAPS[map_idx]
    current_weather_name = WEATHERS[weather_idx][0]

    print(f"Loading map {current_map}...")
    world = client.load_world(current_map)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    world.set_weather(WEATHERS[weather_idx][1])
    blueprint_library = world.get_blueprint_library()

    # Traffic Manager 必須與世界同步，否則 autopilot 車子會停住
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)

    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True, 8000)

    npcs = spawn_npcs(world, blueprint_library)
    print(f"Spawned {len(npcs)} NPC vehicles")

    cam_transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=15.0))
    rgb_queue = queue.Queue()
    sem_queue = queue.Queue()

    rgb_cam, sem_cam = setup_cameras(world, blueprint_library, cam_transform, vehicle, rgb_queue, sem_queue)

    print("Ready! A=Toggle Manual/Auto  Arrow Keys=Drive  R=Record  Q=Quit")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_q:
                        return

                    if event.key == pygame.K_r:
                        recording = not recording
                        if recording:
                            frame_idx = 0
                            print("Recording started!")
                        else:
                            print(f"Recording stopped. {frame_idx} frames saved.")

                    if event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")

                    if event.key == pygame.K_a:
                        manual = not manual
                        vehicle.set_autopilot(not manual)
                        print("Manual ON" if manual else "Autopilot ON")

                    if event.key == pygame.K_w:
                        weather_idx = (weather_idx + 1) % len(WEATHERS)
                        current_weather_name = WEATHERS[weather_idx][0]
                        world.set_weather(WEATHERS[weather_idx][1])
                        print(f"Weather: {current_weather_name}")

                    if event.key == pygame.K_m:
                        print("Switching map...")

                        # 安全清理舊場景
                        try:
                            rgb_cam.stop()
                        except Exception:
                            pass
                        try:
                            sem_cam.stop()
                        except Exception:
                            pass
                        safe_destroy(rgb_cam)
                        safe_destroy(sem_cam)
                        for npc in npcs:
                            safe_destroy(npc)
                        safe_destroy(vehicle)

                        # 清空 queue
                        while not rgb_queue.empty():
                            try:
                                rgb_queue.get_nowait()
                            except Exception:
                                break
                        while not sem_queue.empty():
                            try:
                                sem_queue.get_nowait()
                            except Exception:
                                break

                        # 載入新地圖
                        map_idx = (map_idx + 1) % len(MAPS)
                        current_map = MAPS[map_idx]
                        world = client.load_world(current_map)

                        settings = world.get_settings()
                        settings.synchronous_mode = True
                        settings.fixed_delta_seconds = 0.05
                        world.apply_settings(settings)
                        world.set_weather(WEATHERS[weather_idx][1])

                        blueprint_library = world.get_blueprint_library()
                        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
                        spawn_point = world.get_map().get_spawn_points()[0]
                        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                        vehicle.set_autopilot(not manual)

                        npcs = spawn_npcs(world, blueprint_library)
                        rgb_cam, sem_cam = setup_cameras(
                            world, blueprint_library, cam_transform, vehicle, rgb_queue, sem_queue)

                        print(f"Map loaded: {current_map}")

            if paused:
                clock.tick(30)
                continue

            if manual:
                keys = pygame.key.get_pressed()
                control = get_manual_control(keys)
                vehicle.apply_control(control)

            world.tick()

            rgb_img = rgb_queue.get(timeout=2.0)
            sem_img = sem_queue.get(timeout=2.0)

            rgb_data = np.array(rgb_img.raw_data).reshape((rgb_img.height, rgb_img.width, 4))

            if recording and frame_idx < NUM_FRAMES:
                Image.fromarray(rgb_data[:, :, :3]).save(f"{OUTPUT_RGB}/{frame_idx:06d}.png")

                sem_img.convert(carla.ColorConverter.CityScapesPalette)
                sem_data = np.array(sem_img.raw_data).reshape((sem_img.height, sem_img.width, 4))
                Image.fromarray(sem_data[:, :, :3]).save(f"{OUTPUT_SEMANTIC}/{frame_idx:06d}.png")

                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Recorded {frame_idx}/{NUM_FRAMES} frames")
                if frame_idx >= NUM_FRAMES:
                    recording = False
                    print(f"Done! {NUM_FRAMES} frames saved.")
                    return

            surface = pygame.surfarray.make_surface(rgb_data[:, :, :3].swapaxes(0, 1))
            display.blit(surface, (0, 0))
            draw_ui(display, frame_idx, NUM_FRAMES, recording,
                    current_map, current_weather_name, paused, manual)
            pygame.display.flip()
            clock.tick(60)

    finally:
        try:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        except Exception:
            pass
        try:
            rgb_cam.stop()
            sem_cam.stop()
        except Exception:
            pass
        safe_destroy(rgb_cam)
        safe_destroy(sem_cam)
        for npc in npcs:
            safe_destroy(npc)
        safe_destroy(vehicle)
        pygame.quit()
        print("Recording ended!")


if __name__ == '__main__':
    main()