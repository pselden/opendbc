#!/usr/bin/env python3
"""Closed-loop Odyssey longitudinal sim: the real CarController against a van model fitted from logged drives.

Van model (fitted on 2022 Odyssey routes driven on brikowski ody-op, rmse 0.11 m/s^2, best delay 0.5 s):
  aEgo(t) = (C_GAS * gas(t - 0.5 s) + C0 + C_V2 * v^2) / mass_ratio - B_GRADE * g * sin(pitch)
Force terms scale with 1/mass_ratio; gravity along the road does not. Brake-domain requests are assumed to be
delivered by the car's own brake controller. The planner is a speed-tracking stand-in for openpilot's.

usage: odyssey_load_sim.py           print the summary table (true load x load setting x scenario)
"""
import math
from collections import deque
from unittest.mock import patch

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.odyssey import ODYSSEY_LOAD_FACTORS
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC

DELAY_S = 0.5
C_GAS = 0.00111361
C0 = 0.182683
C_V2 = -0.000344564
B_GRADE = 0.880315

LOADS = {"empty": 1.0, "family": 1.15, "family + luggage": 1.25}
SETTINGS = {"empty": 0, "family": 1, "family + luggage": 2}

# name: (start speed m/s, set speed m/s, grade %, seconds)
SCENARIOS = {
  "level 11->25 m/s": (11.0, 25.0, 0.0, 40.0),
  "4% climb, hold 25": (25.0, 25.0, 4.0, 40.0),
  "4% climb 15->25": (15.0, 25.0, 4.0, 45.0),
  "level cruise 25": (25.0, 25.0, 0.0, 30.0),
}


def planner_accel(v, v_set):
  # stand-in for openpilot's plan: ease toward set speed, capped like a normal personality
  return max(-1.0, min(1.2, 0.35 * (v_set - v)))


def run(scenario, true_load, setting):
  factor = ODYSSEY_LOAD_FACTORS[setting]
  # stand in for the device param so the sim never touches this machine's params
  with patch.object(CarController, "_read_odyssey_load_factor", lambda self: factor):
    return simulate(scenario, true_load)


def simulate(scenario, true_load):
  v0, v_set, grade_pct, seconds = SCENARIOS[scenario]
  CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
  CP_SP = CarInterface.get_params_sp(CP, CP.carFingerprint, gen_empty_fingerprint(), [], True, False, False)
  controller = CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, CP_SP)
  CS = CarState(CP, CP_SP)
  CS.update(CS.get_can_parsers(CP, CP_SP))

  pitch = math.atan(grade_pct / 100.0)
  delay = deque([0.0] * int(round(DELAY_S / DT_CTRL)), maxlen=int(round(DELAY_S / DT_CTRL)))
  v, a, gas = v0, 0.0, 0.0
  steps = int(seconds / DT_CTRL)
  track_err, max_gas, t_reach = [], 0.0, None
  for i in range(steps):
    now = 1_000_000_000 + i * 10_000_000
    request = planner_accel(v, v_set)
    CS.out = structs.CarState(vEgo=v, aEgo=a, canValid=True)
    CS.odyssey_target_gear = 7 if v > 12 else 5
    CS.odyssey_car_gas = 1.0 if gas > 0 else 0.0
    CS.odyssey_computer_braking = False
    CS.odyssey_user_brake = 0.0
    CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., pitch, 0.])
    control.actuators.accel = request
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    output, _ = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
    gas = output.gas
    max_gas = max(max_gas, gas)

    delayed_gas = delay[0]
    delay.append(gas)
    gravity = B_GRADE * ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch)
    if controller.odyssey_brake_selected:
      a = request  # car's brake controller delivers the request
    else:
      a = (C_GAS * delayed_gas + C0 + C_V2 * v * v) / true_load - gravity
    v = max(0.0, v + a * DT_CTRL)
    if i * DT_CTRL >= 3.0:  # skip the first delay window
      track_err.append(request - a)
    if t_reach is None and v >= v_set - 0.5 and v0 < v_set - 0.5:
      t_reach = i * DT_CTRL
  mean_err = sum(track_err) / len(track_err)
  return {"final_v": v, "shortfall": v_set - v, "mean_track_err": mean_err, "max_gas": max_gas, "t_reach": t_reach}


def main():
  print(f"{'scenario':20} {'true load':17} {'setting':17} {'end speed':>9} {'short':>6} {'accel err':>9} {'reach s':>7} {'max gas':>7}")
  for scenario in SCENARIOS:
    for load_name, true_load in LOADS.items():
      for setting_name, setting in SETTINGS.items():
        r = run(scenario, true_load, setting)
        reach = f"{r['t_reach']:.1f}" if r['t_reach'] is not None else "-"
        mark = " <" if setting_name == load_name else ""
        print(f"{scenario:20} {load_name:17} {setting_name:17} {r['final_v']:9.2f} {r['shortfall']:6.2f} " +
              f"{r['mean_track_err']:+9.3f} {reach:>7} {r['max_gas']:7.0f}{mark}")
    print()


if __name__ == "__main__":
  main()
