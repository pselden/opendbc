"""Odyssey response algorithms maintained on top of the MVL Honda integration.

Based on brikowski/opendbc through c91503afe and OdyPilot's van-load adaptation.
"""
import math
from collections import deque

import numpy as np

from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, DT_CTRL
from opendbc.car.common.filter_simple import FirstOrderFilter

ODYSSEY_LOW_SPEED_DOMAIN_VEGO = 5.0
# Include the float32 representation of a nominal -0.10 m/s2 carControl request.
ODYSSEY_GAS_BRIDGE_ENTRY = -0.101
ODYSSEY_GAS_BRIDGE_COMMAND = -60.0
ODYSSEY_GRADE_FILTER_TAU = 0.5
ODYSSEY_GRADE_RAMP_ACCEL = 0.30
ODYSSEY_GRADE_GAIN = 0.6
ODYSSEY_UPHILL_GAS_ACCEL_MAX = 1.0
# comma's Bosch gas map floor; brikowski/opendbc helpers read it from CarControllerParams, which starts at 0.0 in MVL.
ODYSSEY_GAS_MAP_FLOOR = -0.2
ODYSSEY_RESPONSE_ERROR_TAU = 0.3
ODYSSEY_RESPONSE_DELAY_FRAMES = 25  # 0.5 s Bosch longitudinal delay at 50 Hz
ODYSSEY_RESPONSE_MAX_COUNTS = 100.0
ODYSSEY_RESPONSE_COUNTS_PER_ACCEL = 200.0
ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL = 500.0
ODYSSEY_RESPONSE_SLEW_COUNTS = 10.0
ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS = 200.0
ODYSSEY_REQUEST_LOOKBACK_NS = 200_000_000
ODYSSEY_COAST_RELEASE_GRAVITY = 0.25
ODYSSEY_COAST_RELEASE_ERROR = 0.20
# Keep mild negative road-speed requests out of friction braking. Brake selection remains based on
# the raw request; gas continuity and response-qualified coast release never add friction braking.
ODYSSEY_ROAD_BRAKE_ENTRY = -0.30
# Load presets (param OdysseyLoadPreset: 0 empty, 1 family, 2 family + luggage) scale the gas map input by
# the mass ratio to the empty van the controller was tuned on (~4500 lb + driver). Uncalibrated until a
# loaded drive; the panda's gas limit still bounds every preset.
ODYSSEY_LOAD_PRESET_PARAM = "OdysseyLoadPreset"
ODYSSEY_LOAD_FACTORS = (1.0, 1.15, 1.25)
ODYSSEY_LOAD_REFRESH_FRAMES = 200  # re-read the preset every 2 s so a change applies mid-drive
def odyssey_command_domains(accel, speed, previous_brake=False, previous_gas=False, bridge_active=False,
                            gas_accel=None, release_brake=False, release_gas=False, passive_accel=None):
  """Keep low-speed stop authority and separate road-speed coast from friction braking."""
  gas_accel = accel if gas_accel is None else gas_accel
  gas_selected = accel > 0.0
  if speed >= ODYSSEY_LOW_SPEED_DOMAIN_VEGO:
    gas_selected |= (not previous_brake and not previous_gas and ODYSSEY_GAS_BRIDGE_ENTRY <= accel < 0.0 and
                     gas_accel > ODYSSEY_GAS_MAP_FLOOR)
    gas_selected |= previous_gas and min(accel, gas_accel) > ODYSSEY_GAS_MAP_FLOOR
    if bridge_active and accel < ODYSSEY_GAS_BRIDGE_ENTRY:
      gas_selected = False
    brake_selected = not release_brake and (accel < ODYSSEY_ROAD_BRAKE_ENTRY or (previous_brake and accel < 0.0))
    # Holding net acceleration can still need brakes when coasting would accelerate faster.
    if previous_brake and not release_brake and passive_accel is not None and accel < passive_accel:
      gas_selected, brake_selected = False, True
  else:
    brake_selected = accel <= 0.0
  if release_gas:
    gas_selected = False
  # Gas and brake commands remain mutually exclusive.
  return gas_selected, brake_selected and not gas_selected


def odyssey_gas_command(accel, mapped_gas, gas_selected, previous_gas, bridge_active, grade_weight=0.0):
  """Pre-activate gas at a grade-scaled command while keeping the level-road bridge."""
  bridge_active = gas_selected and accel < 0.0 and (bridge_active or not previous_gas)
  gas = ODYSSEY_GAS_BRIDGE_COMMAND + grade_weight * (mapped_gas - ODYSSEY_GAS_BRIDGE_COMMAND) if bridge_active else mapped_gas
  return (gas if gas_selected else 0.0), bridge_active


def odyssey_load_factor(preset):
  """Map a stored load preset to its gas scale; anything unrecognized is the empty van."""
  try:
    index = int(preset)
  except (TypeError, ValueError):
    return 1.0
  return ODYSSEY_LOAD_FACTORS[index] if 0 <= index < len(ODYSSEY_LOAD_FACTORS) else 1.0


def odyssey_load_gas_accel(gas_accel, load_factor):
  """Scale the positive (force-producing) gas map input by vehicle mass; negative input stays unmapped."""
  return gas_accel * load_factor if gas_accel > 0.0 else gas_accel


def odyssey_grade_weight(pitch):
  grade = max(0.0, math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY)
  load = min(grade / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
  return load * load


def odyssey_grade_gas_accel(accel, pitch):
  """Add the grade load before gas calibration and its feedforward ceiling."""
  grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY
  if grade < 0.0:
    # Gravity still supplies acceleration when the requested net acceleration is zero.
    return max(ODYSSEY_GAS_MAP_FLOOR, accel + grade * ODYSSEY_GRADE_GAIN)
  zero_grade = min(grade * ODYSSEY_GRADE_GAIN * odyssey_grade_weight(pitch), ODYSSEY_UPHILL_GAS_ACCEL_MAX)
  if zero_grade > 0.0 and ODYSSEY_ROAD_BRAKE_ENTRY < accel <= 0.0:
    gas_split = ODYSSEY_GAS_MAP_FLOOR
    if accel <= gas_split:
      x = (accel - ODYSSEY_ROAD_BRAKE_ENTRY) / (gas_split - ODYSSEY_ROAD_BRAKE_ENTRY)
      zero_grade *= x * x * (3.0 - 2.0 * x)
    # Preserve request sensitivity through zero, even on a steep climb.
    return accel + zero_grade
  if accel <= 0.0:
    return accel
  x = min(accel / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
  grade_weight = x * x * (3.0 - 2.0 * x)
  grade_accel = zero_grade + (grade * ODYSSEY_GRADE_GAIN - zero_grade) * grade_weight
  return accel + grade_accel


def odyssey_creep_brake_accel(accel, speed):
  """Compensate weak creep-speed braking without retaining a released request."""
  if not math.isfinite(speed) or speed < 0.0 or accel >= 0.0:
    return accel
  correction = float(np.interp(accel, [-0.8, -0.3, -0.1, 0.0], [0.0, -0.25, -0.25, 0.0]))
  weight = float(np.interp(speed, [1.0, 2.0], [1.0, 0.0]))
  return accel + weight * correction


def odyssey_low_speed_gas_command(gas, accel, speed):
  """Bound positive-gas feedforward where Odyssey acceleration exceeds the net request."""
  if accel <= 0.0 or gas <= 0.0:
    return gas

  def smoothstep(value):
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

  speed_weight = smoothstep((speed - 8.0) / 4.0) * smoothstep((24.0 - speed) / 4.0)
  request_weight = smoothstep((accel - 0.4) / 0.4) * smoothstep((2.0 - accel) / 0.4)
  return max(0.0, gas - ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS * speed_weight * request_weight)


def odyssey_gas_feedforward(accel, gas_accel, speed, params, load_factor=1.0):
  """Apply gas calibration and its uphill ceiling together, before response feedback."""
  def calibrated(map_accel, request):
    gas = float(np.interp(odyssey_load_gas_accel(map_accel, load_factor), params.BOSCH_GAS_LOOKUP_BP, params.BOSCH_GAS_LOOKUP_V))
    return odyssey_low_speed_gas_command(gas, request, speed)

  gas = calibrated(gas_accel, accel)
  if gas_accel <= accel:
    return gas, 0.0
  ceiling_request = max(accel, ODYSSEY_UPHILL_GAS_ACCEL_MAX)
  capped = min(gas, calibrated(ceiling_request, ceiling_request))
  cap_weight = 0.0
  if accel > 0.0 and capped < gas:
    base = calibrated(accel, accel)
    if gas > base:
      limited_fraction = float(np.clip((gas - capped) / (gas - base), 0.0, 1.0))
      load = min((gas_accel - accel) / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
      cap_weight = limited_fraction * load * load * (3.0 - 2.0 * load)
  return capped, cap_weight


class OdysseyGasResponse:
  """Compare delayed gas requests and response within the same received drivetrain state."""
  def __init__(self):
    self.correction = 0.0
    self.integral = 0.0
    self.requests = deque(maxlen=ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
    self.pitch_peak = None
    self.drivetrain = None
    self.paused = False
    self.error = FirstOrderFilter(0.0, ODYSSEY_RESPONSE_ERROR_TAU, DT_CTRL * 2, initialized=False)

  def reset(self):
    self.__init__()

  def clear_observer(self):
    self.integral = 0.0
    self.requests.clear()
    self.pitch_peak = None
    self.paused = False
    self.error = FirstOrderFilter(0.0, ODYSSEY_RESPONSE_ERROR_TAU, DT_CTRL * 2, initialized=False)

  def update(self, request, aego, pitch, speed, gear, eligible, cap_weight=0.0, shift_activity=0,
             limits=(-ODYSSEY_RESPONSE_MAX_COUNTS, ODYSSEY_RESPONSE_MAX_COUNTS), coast=False):
    if (not (eligible or coast) or not all(math.isfinite(v) for v in (request, aego, pitch, speed, gear, cap_weight, *limits)) or
        not 1 <= gear <= 10):
      self.reset()
      return 0.0

    drivetrain = (gear, shift_activity)
    if coast:
      # Coast supplies no gas-response samples; retain only a valid learned gas reduction.
      offset = self.integral
      if drivetrain != self.drivetrain or offset >= 0.0:
        self.reset()
      else:
        self.clear_observer()
        self.integral = offset
        self.paused = True
      return 0.0
    if self.paused:
      if drivetrain != self.drivetrain:
        self.reset()
      else:
        # This offset was not issued during coast; rebase it to the current gas headroom.
        self.correction = float(np.clip(self.integral, *limits))
        self.paused = False
    if drivetrain != self.drivetrain:
      self.clear_observer()
    elif self.pitch_peak is not None and pitch < self.pitch_peak - 0.01:
      # Falling grade can invalidate added gas, but should not erase overacceleration feedback.
      if self.integral > 0.0:
        self.clear_observer()
      self.pitch_peak = pitch
    self.drivetrain = drivetrain
    self.pitch_peak = pitch if self.pitch_peak is None else max(self.pitch_peak, pitch)
    self.requests.append((request, cap_weight))
    # Fresh feedback and warmup use the mapped headroom.
    lower = limits[0]
    upper = limits[1]
    target = 0.0
    if len(self.requests) > ODYSSEY_RESPONSE_DELAY_FRAMES:
      delayed_request, delayed_cap_weight = self.requests[0]
      response_gain = ODYSSEY_RESPONSE_COUNTS_PER_ACCEL + (ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL -
                                                          ODYSSEY_RESPONSE_COUNTS_PER_ACCEL) * float(np.clip(delayed_cap_weight, 0.0, 1.0))
      residual = self.error.update(delayed_request - aego)
      # Delayed feedback must agree with the latest acceleration error.
      residual = float(np.clip(residual, min(request - aego, 0.0), max(request - aego, 0.0)))
      proportional = response_gain * residual
      integral = float(np.clip(self.integral + ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * residual * DT_CTRL * 2,
                                lower, upper))
      target = proportional + integral
      # Retain the offset at zero error without accumulating beyond usable gas authority.
      if lower <= target <= upper or (target > upper and residual < 0.0) or (target < lower and residual > 0.0):
        self.integral = integral
      target = float(np.clip(proportional + self.integral, lower, upper))
      if target > 0.0 and residual < -ODYSSEY_COAST_RELEASE_ERROR:
        # Withdraw added gas on qualified excess without discarding the learned offset.
        target = min(target, max(0.0, self.correction + proportional))
    else:
      # Observed excess acceleration can reduce gas before the delayed observer is ready.
      proportional = float(np.clip(ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * min(request - aego, 0.0), lower, 0.0))
      target = float(np.clip(self.correction, lower, upper))
      # Scale withdrawal of an issued correction to the current reverse error.
      if (request - aego) * target < 0.0:
        target = float(np.clip(target + ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * (request - aego), lower, max(target, 0.0)))
      if request < aego:
        target = min(target, max(self.correction, 0.0) + proportional)
      # Seed the offset after accounting for the current proportional reduction.
      self.integral = target - proportional
    self.correction = float(np.clip(target, self.correction - ODYSSEY_RESPONSE_SLEW_COUNTS,
                                     self.correction + ODYSSEY_RESPONSE_SLEW_COUNTS))
    return self.correction


def odyssey_brake_release(now_nanos, request, aego, previous_brake, braking_ts_nanos, car_gas, gear, eligible, passive_accel=None,
                          coast_from_brake=False):
  """Prefer qualified coast when it can supply the requested deceleration."""
  if (not eligible or passive_accel is None or not all(math.isfinite(v) for v in (request, aego, car_gas, gear, passive_accel)) or
      not 1 <= gear <= 10 or car_gas != 0 or
      braking_ts_nanos <= 0 or not 0 <= now_nanos - braking_ts_nanos < 60_000_000):
    return False
  # Require excess braking to release; tolerate the existing response margin for a qualified handoff.
  margin = -ODYSSEY_COAST_RELEASE_ERROR if previous_brake else ODYSSEY_COAST_RELEASE_ERROR
  forecast_margin = ODYSSEY_COAST_RELEASE_ERROR if coast_from_brake and aego <= request else 0.0
  return request < 0.0 and passive_accel <= request + forecast_margin and aego < request + margin


class OdysseyCoastResponse:
  """Return a prior passive estimate, or None if unlearned or currently ineligible."""
  def __init__(self):
    self.drag = {}
    self.samples = deque(maxlen=5)
    self.gear = None
    self.coast_start = None

  def reset(self):
    self.__init__()

  def update(self, frame, gear, pitch, aego, coast, eligible):
    if not all(math.isfinite(v) for v in (gear, pitch, aego)):
      self.reset()
      return None
    if not eligible or not 1 <= gear <= 10:
      # Idle stop pauses forward-gear observations; keep completed calibration.
      self.gear = None
      self.coast_start = None
      self.samples.clear()
      return None
    if gear != self.gear:
      self.gear = gear
      self.coast_start = None

    gravity = -ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch)
    forecast = gravity + self.drag[gear] if gear in self.drag else None
    if not coast:
      self.coast_start = None
    else:
      if self.coast_start is None:
        self.coast_start = frame
        # Median observations belong to one uninterrupted coast.
        self.samples.clear()
      # Acquire the first five settled samples promptly; retain the learned update cadence.
      if frame - self.coast_start >= 50 and (gear not in self.drag or frame % 10 == 0):
        self.samples.append(aego - gravity)
        if len(self.samples) == self.samples.maxlen:
          self.drag[gear] = sorted(self.samples)[2]
    return forecast


class OdysseyGasCoastRelease:
  """Release excess gas into coast using prior coast response or qualified gas exhaustion."""
  def __init__(self):
    self.requests = deque(maxlen=12)
    self.gear = None
    self.active = False
    self.exhausted = False

  def reset(self):
    self.__init__()

  def update(self, now_nanos, request, aego, speed, pitch, gear, passive_accel, previous_gas, eligible, gas_floor=False,
             floor_excess=False, downhill=True):
    exhausted_gas = previous_gas and gas_floor and floor_excess
    if (not eligible or (not downhill and not exhausted_gas and not self.exhausted) or not 1 <= gear <= 10 or
        not all(math.isfinite(v) for v in (request, aego, speed, pitch)) or
        (passive_accel is not None and not math.isfinite(passive_accel))):
      self.reset()
      return False
    if gear != self.gear:
      self.reset()
      self.gear = gear
    self.requests.append((now_nanos, request))
    prior = next((sample for sample in self.requests if sample[0] <= now_nanos - ODYSSEY_REQUEST_LOOKBACK_NS), None)
    request_slope = (request - prior[1]) / ((now_nanos - prior[0]) * 1e-9) if prior is not None else 0.0

    downhill_gravity = -ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch)
    # A prior forecast must not undo exhausted-gas coast while measured excess remains.
    if (request <= ODYSSEY_ROAD_BRAKE_ENTRY or
        (passive_accel is not None and passive_accel - request < 0.05 and
         not self.exhausted) or
        not ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= speed <= 35.0 or
        (downhill_gravity <= ODYSSEY_COAST_RELEASE_GRAVITY and not self.exhausted) or
        (aego - request < 0.05 and not
         (passive_accel is not None and passive_accel - request >= 0.05 and
          aego >= request - ODYSSEY_COAST_RELEASE_ERROR))):
      self.active = self.exhausted = False
    if (not self.active and (previous_gas or passive_accel is not None) and ODYSSEY_LOW_SPEED_DOMAIN_VEGO <= speed <= 35.0 and
        ODYSSEY_ROAD_BRAKE_ENTRY < request and
        aego - request > ODYSSEY_COAST_RELEASE_ERROR and
        (exhausted_gas or
         (downhill and downhill_gravity > ODYSSEY_COAST_RELEASE_GRAVITY and
          ((passive_accel is not None and passive_accel - request > ODYSSEY_COAST_RELEASE_ERROR) or
           (passive_accel is None and request < 0.0 and gas_floor and request_slope < -0.20))))):
      self.active = True
      self.exhausted = exhausted_gas
    return self.active
