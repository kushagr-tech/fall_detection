"""
test_sensor_scenarios.py
========================
Tests real Android emulator sensor streams via `adb emu sensor set`:
1. Normal Activity (Walking): verifies no false alarms.
2. Phone Drop: mechanical spike without biological fall dynamics or lying posture.
3. Genuine Fall: free-fall dip -> rotation -> impact -> post-fall lying posture.
"""

import time
import subprocess
import math

ADB = "/Users/kushagragarwal/Library/Android/sdk/platform-tools/adb"

def set_sensors(ax, ay, az, gx=0.0, gy=0.0, gz=0.0):
    cmd_acc = f"{ADB} emu sensor set acceleration {ax:.3f}:{ay:.3f}:{az:.3f}"
    cmd_gyr = f"{ADB} emu sensor set gyroscope {gx:.3f}:{gy:.3f}:{gz:.3f}"
    subprocess.run(cmd_acc, shell=True, check=True)
    subprocess.run(cmd_gyr, shell=True, check=True)

def simulate_normal_walking(duration_sec=6.0):
    print(f"\n[Test 1] Simulating Normal Walking for {duration_sec}s...")
    start = time.time()
    t = 0.0
    while (time.time() - start) < duration_sec:
        # Walking cadence ~ 1.8 Hz
        # Gravity on Y axis (portrait) with vertical bounce and lateral sway
        ay = 9.81 + 3.2 * math.sin(2 * math.pi * 1.8 * t)
        ax = 1.2 * math.sin(2 * math.pi * 0.9 * t)
        az = 0.8 * math.cos(2 * math.pi * 1.8 * t)
        gx = 0.4 * math.sin(2 * math.pi * 1.8 * t)
        gy = 0.3 * math.cos(2 * math.pi * 0.9 * t)
        gz = 0.2 * math.sin(2 * math.pi * 1.8 * t)
        set_sensors(ax, ay, az, gx, gy, gz)
        t += 0.02
        time.sleep(0.02)
    print("✓ Normal walking completed.")

def simulate_phone_drop():
    print("\n[Test 2] Simulating Phone Drop...")
    # Upright standing initially
    set_sensors(0.0, 9.81, 0.0, 0.0, 0.0, 0.0)
    time.sleep(1.0)
    
    # Free-fall 300 ms
    for _ in range(15):
        set_sensors(0.2, 0.5, 0.1, 0.5, 0.2, 0.3)
        time.sleep(0.02)
        
    # Hard impact spike 40 ms (sharp rigid collision)
    set_sensors(25.0, 35.0, 20.0, 4.0, 3.5, 2.0)
    time.sleep(0.04)
    
    # User immediately picks up phone or keeps moving
    print("User moves/picks up phone (resumed active motion)...")
    for i in range(50):
        t = i * 0.02
        ay = 9.81 + 4.0 * math.sin(2 * math.pi * 1.5 * t)
        ax = 2.0 * math.cos(2 * math.pi * 1.5 * t)
        set_sensors(ax, ay, 1.0, 1.5, 1.2, 0.8)
        time.sleep(0.02)
    print("✓ Phone drop simulation completed.")

def simulate_controlled_fall():
    print("\n[Test 3] Simulating Controlled Real-World Fall...")
    # 1. Normal upright standing for 1s
    print("1. Upright baseline...")
    for _ in range(50):
        set_sensors(0.1, 9.75, 0.8, 0.0, 0.0, 0.0)
        time.sleep(0.02)
        
    # 2. Free fall weightlessness dip + tumble rotation (400ms)
    print("2. Loss of balance / tumble...")
    for _ in range(20):
        set_sensors(1.0, 3.2, 1.5, 2.8, 2.2, 2.5)
        time.sleep(0.02)
        
    # 3. Heavy impact spike (250ms)
    print("3. Heavy ground impact...")
    for _ in range(12):
        set_sensors(14.0, 24.5, 12.0, 3.5, 2.8, 2.2)
        time.sleep(0.02)
        
    # 4. Post-fall lying still on ground (posture rotated to Z axis, zero variance)
    print("4. Post-fall lying still on floor...")
    for _ in range(150): # 3 seconds
        set_sensors(0.1, 0.2, 9.81, 0.02, 0.01, 0.01)
        time.sleep(0.02)
    print("✓ Controlled fall sequence completed.")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("walk", "all"):
        simulate_normal_walking()
    if mode in ("drop", "all"):
        simulate_phone_drop()
    if mode in ("fall", "all"):
        simulate_controlled_fall()
