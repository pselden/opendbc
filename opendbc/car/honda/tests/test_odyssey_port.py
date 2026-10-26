# Odyssey command-following tests ported from brikowski/opendbc (opendbc/car/honda/tests/test_honda.py @ 164b2fd).
# Adapted to sunnypilot's CarParamsSP / CarControlSP API; test bodies are unchanged.
import math
import subprocess
import sys
import unittest
from collections import deque
from itertools import product
from unittest.mock import patch

from opendbc.can import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, DT_CTRL, Bus, gen_empty_fingerprint, structs
from opendbc.car.honda.carcontroller import (ODYSSEY_GAS_BRIDGE_COMMAND, ODYSSEY_GRADE_GAIN,
                                             ODYSSEY_GRADE_RAMP_ACCEL, ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS,
                                             ODYSSEY_ZERO_GAS_GRADE_GAIN,
                                             ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL, ODYSSEY_RESPONSE_COUNTS_PER_ACCEL,
                                             ODYSSEY_RESPONSE_DELAY_FRAMES, ODYSSEY_RESPONSE_MAX_COUNTS, ODYSSEY_RESPONSE_SLEW_COUNTS,
                                             ODYSSEY_ROAD_BRAKE_ENTRY,
                                             ODYSSEY_UPHILL_GAS_ACCEL_MAX, ODYSSEY_LOAD_FACTORS, CarController,
                                             odyssey_brake_release, odyssey_load_factor, odyssey_load_gas_accel,
                                             OdysseyCoastResponse, OdysseyGasCoastRelease, OdysseyGasResponse,
                                             odyssey_command_domains, odyssey_gas_command, odyssey_grade_weight,
                                             odyssey_creep_brake_accel, odyssey_low_speed_gas_command,
                                             odyssey_grade_gas_accel, odyssey_gas_feedforward, ODYSSEY_GAS_MAP_FLOOR,
)
from opendbc.car.honda.values import CAR, CarControllerParams
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import DBC


def _sp(CP):
  return CarInterface.get_params_sp(CP, CP.carFingerprint, gen_empty_fingerprint(), [], CP.openpilotLongitudinalControl, False, False)


from types import SimpleNamespace as _SimpleNamespace


class SimpleNamespace(_SimpleNamespace):
  """Fake CarState that falls back to a real sunnypilot Honda CarState for attributes MVL's controller reads."""
  _defaults = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    if name == 'steer_control_active':
      return True  # EPS accepting commands, as when the car is steering
    if SimpleNamespace._defaults is None:
      CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
      SimpleNamespace._defaults = CarState(CP, _sp(CP))
    return getattr(SimpleNamespace._defaults, name)

class TestOdysseySteeringRange(unittest.TestCase):
  def test_parameter_rescaling_preserves_ordinary_physical_map(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    params = CarControllerParams(CP)
    self.assertEqual(params.STEER_MAX, 3840)
    self.assertEqual(params.STEER_DELTA_UP * params.STEER_MAX, 3 * 2560)
    self.assertAlmostEqual(CP.lateralTuning.torque.latAccelFactor, 1.35, places=6)
    self.assertAlmostEqual(CP.lateralTuning.torque.friction, 0.2 / 1.5, places=6)
    for demand in (0.2, 0.5, 0.8):
      old_count = (demand + 0.2 * 0.9) / 0.9 * 2560
      new_count = (demand + CP.lateralTuning.torque.friction * CP.lateralTuning.torque.latAccelFactor) / CP.lateralTuning.torque.latAccelFactor * 3840
      self.assertAlmostEqual(old_count, new_count, delta=0.01)

  def test_steering_wire_preserves_persistent_request_through_ramp_and_turn_exit(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    for direction, alpha_long, bus in ((-1, True, 1), (1, True, 1), (-1, False, 0), (1, False, 0)):
      with self.subTest(direction=direction, alpha_long=alpha_long):
        CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], alpha_long, False, False)
        dbc = DBC[CP.carFingerprint][Bus.pt]
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        parser = CANParser(dbc, [('STEERING_CONTROL', 0)], bus)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        control = structs.CarControl()
        control.enabled = control.latActive = True
        control.actuators.torque = -direction * 0.9
        state = structs.CarState()
        state.vEgo = 22.0
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=math.nan,
                             odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=0,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
        counts = []
        for frame in range(200):
          demand = 1.4 if frame < 100 else 1.25
          control.actuators.curvature = direction * demand / state.vEgo**2
          control.currentCurvature = direction * (demand - 0.4) / state.vEgo**2
          if frame >= 160:
            control.currentCurvature = control.actuators.curvature
          if frame >= 175:
            control.actuators.torque = -direction * 0.5
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, frame * 10_000_000)
          steering = [(addr, dat, src) for addr, dat, src in sends if addr == 0xE4 and src == bus]
          self.assertEqual(len(steering), 1)
          for addr, dat, src in steering:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, src, dat)))
          parser.update([(frame * 10_000_000, steering)])
          counts.append(parser.vl['STEERING_CONTROL']['STEER_TORQUE'])
          self.assertEqual(counts[-1], output.torqueOutputCan)
          self.assertLessEqual(abs(counts[-1]), 3840)
          if 100 <= frame < 175:
            self.assertAlmostEqual(counts[-1], counts[99], delta=1)
        self.assertGreater(abs(counts[34]), 2560)
        self.assertGreater(abs(counts[99]), 3000)
        self.assertEqual(abs(counts[-1]), 1920)
        self.assertLessEqual(max(abs(b - a) for a, b in zip(counts, counts[1:], strict=False)), 77)

  def test_full_range_at_every_speed_in_both_bus_modes(self):
    # brikowski ody-op-test cef93020: no speed or tracking-error gate; the whole 3840 range is reachable
    # at any speed, at the stock slew in counts per frame, and only while lateral is active.
    for alpha_long, bus in ((True, 1), (False, 0)):
      for speed in (3.0, 8.5, 14.9, 22.0, 31.0, 33.1, 38.0):
        with self.subTest(alpha_long=alpha_long, speed=speed):
          CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], alpha_long, False, False)
          dbc = DBC[CP.carFingerprint][Bus.pt]
          controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
          parser = CANParser(dbc, [('STEERING_CONTROL', 0)], bus)
          control = structs.CarControl()
          control.enabled = True
          control.latActive = True
          control.actuators.torque = -1.0
          control.actuators.curvature = 1.8 / speed**2
          control.currentCurvature = control.actuators.curvature
          state = structs.CarState()
          state.vEgo = speed
          CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                               odyssey_car_gas=math.nan,
                               odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=0,
                               odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

          def step(frame, controller=controller, control=control, CS=CS, bus=bus, parser=parser):
            output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, frame * 10_000_000)
            steering = [(addr, dat, src) for addr, dat, src in sends if addr == 0xE4]
            self.assertEqual(len(steering), 1)
            self.assertEqual(steering[0][2], bus)
            parser.update([(frame * 10_000_000, steering)])
            self.assertEqual(parser.vl['STEERING_CONTROL']['STEER_TORQUE'], output.torqueOutputCan)
            self.assertAlmostEqual(output.torque, -output.torqueOutputCan / 3840, places=6)
            return output.torqueOutputCan

          counts = [step(frame) for frame in range(80)]
          # Fractional CAN counts must survive internally between frames.
          for frame, value in enumerate(counts):
            expected = min((frame + 1) * 2 * DT_CTRL * 3840, 3840)
            self.assertLessEqual(abs(value - expected), 1.000001)
          self.assertLessEqual(max(abs(right - left) for left, right in zip(counts, counts[1:], strict=False)), 77)
          self.assertEqual(counts[-1], 3840)
          control.actuators.torque = -1.5  # out-of-range requests saturate at the range, never beyond
          self.assertEqual(step(80), 3840)
          control.actuators.torque = 1.0
          reverse = [step(frame) for frame in range(81, 190)]
          self.assertLessEqual(max(abs(right - left) for left, right in zip(reverse, reverse[1:], strict=False)), 77)
          self.assertEqual(reverse[-1], -3840)
          control.latActive = False
          self.assertEqual(step(190), 0)


class TestOdysseyLongitudinal(unittest.TestCase):
  def test_gas_entry_waits_for_received_brake_release(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for dwell in (2, 40):
      for feedback in ('computer_braking', 'computer_braking_nan', 'user_brake'):
        with self.subTest(dwell=dwell, feedback=feedback):
          controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
          controller.odyssey_brake_selected = True
          CS = CarState(CP, _sp(CP))
          CS.update(CS.get_can_parsers(CP, _sp(CP)))
          CS.out = structs.CarState(vEgo=20., aEgo=-.3, canValid=True)
          CS.odyssey_target_gear = 7
          CS.odyssey_car_gas = 0.
          CS.odyssey_computer_braking = feedback.startswith('computer_braking')
          CS.odyssey_user_brake = .125 if feedback == 'user_brake' else math.nan if feedback == 'computer_braking_nan' else 0.
          control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
          control.actuators.accel = .05
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
          parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)

          def step(controller=controller, CS=CS, control=control, parser=parser):
            now = 1_000_000_000 + controller.frame * 10_000_000
            CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
            output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
            parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
            return output, parser.vl['ACC_CONTROL']

          for _ in range(dwell):
            output, signals = step()
            self.assertEqual(signals['BRAKE_REQUEST'], 0)
            self.assertEqual(signals['GAS_COMMAND'], -30000)
            self.assertEqual(output.gas, 0.)
            self.assertAlmostEqual(signals['ACCEL_COMMAND'], .05, delta=.01)
          control.actuators.accel = -.5
          for _ in range(2):
            _, signals = step()
          self.assertEqual(signals['BRAKE_REQUEST'], 1)
          self.assertEqual(signals['GAS_COMMAND'], -30000)
          control.actuators.accel = .05
          CS.odyssey_computer_braking = False
          CS.odyssey_user_brake = .015625
          for _ in range(2):
            output, signals = step()
          self.assertGreater(signals['GAS_COMMAND'], 0)
          self.assertEqual(signals['BRAKE_REQUEST'], 0)
          self.assertAlmostEqual(signals['ACCEL_COMMAND'], .05, delta=.01)

  def test_gas_entry_does_not_wait_without_reliable_active_braking(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for fault in ('idle', 'negative_feedback', 'nan_feedback', 'stale', 'future', 'missing',
                  'invalid_can', 'invalid_pose', 'low_speed', 'high_speed', 'already_gas'):
      with self.subTest(fault=fault):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.odyssey_brake_selected = True
        controller.odyssey_gas_selected = fault == 'already_gas'
        CS = CarState(CP, _sp(CP))
        CS.update(CS.get_can_parsers(CP, _sp(CP)))
        speed = 7.9 if fault == 'low_speed' else 35.1 if fault == 'high_speed' else 20.
        CS.out = structs.CarState(vEgo=speed, aEgo=-.3, canValid=fault != 'invalid_can')
        now = 1_000_000_000
        CS.odyssey_target_gear = 7
        CS.odyssey_car_gas = 0.
        CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = now - 5_000_000
        CS.odyssey_computer_braking_ts_nanos = {'stale': now - 60_000_000, 'future': now + 1,
                                               'missing': 0}.get(fault, now - 5_000_000)
        CS.odyssey_computer_braking = fault not in ('idle', 'negative_feedback', 'nan_feedback')
        CS.odyssey_user_brake = {'negative_feedback': -.125, 'nan_feedback': math.nan}.get(fault, 0.)
        control = structs.CarControl(enabled=True, longActive=True,
                                     orientationNED=[] if fault == 'invalid_pose' else [0., 0., 0.])
        control.actuators.accel = .05
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        self.assertGreater(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], .05, delta=.01)

  def test_creep_braking_is_bounded_monotone_and_releases_with_request(self):
    for speed in (0.0, 0.5, 1.0, 1.5, 2.0, 5.0):
      requests = [-3.5 + i * 0.005 for i in range(741)]
      commands = [odyssey_creep_brake_accel(q, speed) for q in requests]
      self.assertTrue(all(left <= right + 1e-12 for left, right in zip(commands, commands[1:], strict=False)))
      for q, command in zip(requests, commands, strict=True):
        self.assertLessEqual(q - command, 0.25 + 1e-12)
        self.assertGreaterEqual(q - command, -1e-12)
        if q <= -0.8 or q >= 0.0 or speed >= 2.0:
          self.assertAlmostEqual(command, q)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.3, 0.6), -0.55)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.15, 0.6), -0.40)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.1, 0.6), -0.35)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.05, 0.6), -0.175)
    self.assertAlmostEqual(odyssey_creep_brake_accel(-0.3, 1.5), -0.425)
    for speed in (-1.0, math.nan, math.inf):
      self.assertEqual(odyssey_creep_brake_accel(-0.3, speed), -0.3)
    for request in (-0.8, -0.3, -0.1, 0.0):
      self.assertLess(abs(odyssey_creep_brake_accel(request - 1e-6, 0.5) -
                          odyssey_creep_brake_accel(request + 1e-6, 0.5)), 8e-6)

  def test_nonfinite_pitch_falls_back_to_raw_command_and_recovers(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request in (-0.6, 0.02, 0.6):
      for invalid_pitch in (math.nan, math.inf, -math.inf):
        with self.subTest(request=request, invalid_pitch=invalid_pitch):
          controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
          parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
          control = structs.CarControl()
          control.enabled = True
          control.longActive = True
          control.actuators.accel = request
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
          state = structs.CarState()
          state.vEgo = 16.0
          CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                               odyssey_car_gas=math.nan,
                               odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=3,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

          def step(pitch, control=control, controller=controller, CS=CS, parser=parser):
            control.orientationNED = [0.0, pitch, 0.0]
            acc = []
            for _ in range(2):
              frame = controller.frame
              _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, frame * 10_000_000)
              acc.extend((addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1)
            self.assertEqual(len(acc), 1)
            parser.update([(frame * 10_000_000, acc)])
            return parser.vl['ACC_CONTROL']

          step(0.04)
          invalid = step(invalid_pitch)
          self.assertAlmostEqual(invalid['ACCEL_COMMAND'], request, delta=0.01)
          self.assertEqual(invalid['BRAKE_REQUEST'], request < 0.0)
          if request > 0.0:
            raw_gas = float((request - ODYSSEY_GAS_MAP_FLOOR) /
                            (CarControllerParams.BOSCH_GAS_LOOKUP_BP[-1] - ODYSSEY_GAS_MAP_FLOOR) *
                            CarControllerParams.BOSCH_GAS_LOOKUP_V[-1])
            self.assertAlmostEqual(invalid['GAS_COMMAND'], odyssey_low_speed_gas_command(raw_gas, request, state.vEgo), delta=1)
          restored = step(0.04)
          self.assertTrue(math.isfinite(controller.odyssey_pitch.x))
          self.assertTrue(math.isfinite(restored['ACCEL_COMMAND']))

  def test_received_engine_torque_is_parsed_from_odyssey_powertrain_bus(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    parser = CarState(CP, _sp(CP)).get_can_parsers(CP, _sp(CP))[Bus.pt]
    dbc = DBC[CP.carFingerprint][Bus.pt]
    packer = CANPacker(dbc)
    now_nanos = 1_000_000_000
    parser.update([(now_nanos, [packer.make_can_msg('GAS_PEDAL_2', parser.bus,
                                                     {'ENGINE_TORQUE_ESTIMATE': -145, 'CAR_GAS': 0})])])
    self.assertEqual(parser.vl['GAS_PEDAL_2']['ENGINE_TORQUE_ESTIMATE'], -145)
    self.assertEqual(parser.vl['GAS_PEDAL_2']['CAR_GAS'], 0)
    self.assertEqual(parser.ts_nanos['GAS_PEDAL_2']['ENGINE_TORQUE_ESTIMATE'], now_nanos)

  def test_low_speed_positive_gas_trim_is_bounded_and_monotone(self):
    def gas(accel):
      return (accel + 0.2) / 2.2 * 2000.0
    self.assertEqual(ODYSSEY_LOW_SPEED_GAS_TRIM_COUNTS, 200.0)
    self.assertEqual(odyssey_low_speed_gas_command(gas(1.1), 1.1, 14.0), gas(1.1) - 200.0)
    for speed in (0.0, 8.0, 24.0, 32.0):
      self.assertEqual(odyssey_low_speed_gas_command(gas(1.1), 1.1, speed), gas(1.1))
    for accel in (0.4, 2.0):
      self.assertEqual(odyssey_low_speed_gas_command(gas(accel), accel, 14.0), gas(accel))
    self.assertEqual(odyssey_low_speed_gas_command(-60.0, -0.1, 14.0), -60.0)
    self.assertEqual(odyssey_low_speed_gas_command(0.0, 0.0, 14.0), 0.0)
    values = [odyssey_low_speed_gas_command(gas(accel), accel, 14.0) for accel in (i * 0.01 for i in range(201))]
    self.assertTrue(all(left <= right for left, right in zip(values, values[1:], strict=False)))
    for speed in (8.0, 12.0, 20.0, 24.0):
      self.assertLess(abs(odyssey_low_speed_gas_command(gas(1.1), 1.1, speed - 1e-4) -
                          odyssey_low_speed_gas_command(gas(1.1), 1.1, speed + 1e-4)), 0.01)

  def test_larger_positive_request_does_not_reduce_gas_in_the_same_vehicle_state(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for eco in (False, True):
      with self.subTest(eco=eco):
        commands = []
        for request in (.50, .52, .54, .56, .58, .60, .62, .64, .66, .68, .70, .72, .74, .76, .78, .80, .90, 1.0):
          controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
          controller.odyssey_pitch.x = .04
          control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., .04, 0.])
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
          state = structs.CarState(vEgo=14., aEgo=1., canValid=True)
          CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                               odyssey_car_gas=0., odyssey_target_gear=7, odyssey_shift_activity=119,
                               odyssey_user_brake=0., odyssey_computer_braking=False)
          parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
          for frame in range(65):
            control.actuators.accel = request if frame == 64 else .65
            now = 1_000_000_000 + frame * 10_000_000
            CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
            _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
            parser.update([(now, sends)])
          signals = parser.vl['ACC_CONTROL']
          self.assertEqual(signals['BRAKE_REQUEST'], 0)
          self.assertAlmostEqual(signals['ACCEL_COMMAND'], request, delta=.01)
          commands.append(signals['GAS_COMMAND'])
        self.assertEqual(commands, sorted(commands))

  def test_gas_response_gain_context_follows_its_delayed_request(self):
    warmup, capped_warmup = OdysseyGasResponse(), OdysseyGasResponse()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      plain = warmup.update(.2, .24, .04, 19., 7, True, limits=(-600., 600.))
      extra = capped_warmup.update(.2, .24, .04, 19., 7, True, cap_weight=1., limits=(-600., 600.))
      self.assertEqual(extra, plain)
    baseline, changed = OdysseyGasResponse(), OdysseyGasResponse()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 1):
      for response in (baseline, changed):
        response.update(.95, .65, .04, 19., 7, True, limits=(-600., 600.))
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      plain = baseline.update(.95, .65, .04, 19., 7, True, limits=(-600., 600.))
      extra = changed.update(.95, .65, .04, 19., 7, True, cap_weight=1., limits=(-600., 600.))
      self.assertEqual(extra, plain)
    plain = baseline.update(.95, .65, .04, 19., 7, True, limits=(-600., 600.))
    extra = changed.update(.95, .65, .04, 19., 7, True, cap_weight=1., limits=(-600., 600.))
    self.assertGreater(extra, plain)

  def test_grade_gas_load_is_continuous_through_negative_transition(self):
    pitch = 0.03
    grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY
    load = min(grade / ODYSSEY_GRADE_RAMP_ACCEL, 1.0)
    zero_grade = grade * ODYSSEY_ZERO_GAS_GRADE_GAIN * load * load
    self.assertAlmostEqual(odyssey_grade_gas_accel(0.0, pitch), zero_grade)
    self.assertGreater(odyssey_grade_gas_accel(-0.10, pitch), -0.10)
    self.assertAlmostEqual(odyssey_grade_gas_accel(-0.20, pitch), -0.20 + zero_grade)
    self.assertEqual(odyssey_grade_gas_accel(ODYSSEY_ROAD_BRAKE_ENTRY, pitch), ODYSSEY_ROAD_BRAKE_ENTRY)
    self.assertEqual(odyssey_grade_gas_accel(0.3, 0.0), 0.3)
    downhill_grade = math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.6
    self.assertAlmostEqual(odyssey_grade_gas_accel(0.3, -pitch), 0.3 - downhill_grade)
    self.assertAlmostEqual(odyssey_grade_gas_accel(0.01, pitch) - 0.01, zero_grade, delta=0.001)
    expected = ODYSSEY_GRADE_RAMP_ACCEL + math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * 0.6
    self.assertEqual(ODYSSEY_GRADE_GAIN, 0.6)
    self.assertAlmostEqual(odyssey_grade_gas_accel(ODYSSEY_GRADE_RAMP_ACCEL, pitch), expected)
    self.assertAlmostEqual(odyssey_grade_gas_accel(1.2, pitch), 1.2 + downhill_grade)
    self.assertAlmostEqual(odyssey_grade_gas_accel(1.2, -pitch), 1.2 - downhill_grade)

  def test_downhill_gas_load_remains_present_through_zero_request(self):
    for pitch in (-0.01, -0.03, -0.08):
      with self.subTest(pitch=pitch):
        zero = odyssey_grade_gas_accel(0.0, pitch)
        self.assertLess(zero, -0.05)
        self.assertGreaterEqual(zero, ODYSSEY_GAS_MAP_FLOOR)
        requests = [-0.3 + i * 0.001 for i in range(901)]
        mapped = [odyssey_grade_gas_accel(q, pitch) for q in requests]
        self.assertTrue(all(left <= right for left, right in zip(mapped, mapped[1:], strict=False)))
        for request in (0.0, 0.3, ODYSSEY_GAS_MAP_FLOOR - math.sin(pitch) * ACCELERATION_DUE_TO_GRAVITY * ODYSSEY_GRADE_GAIN):
          self.assertAlmostEqual(odyssey_grade_gas_accel(request - 1e-6, pitch),
                                 odyssey_grade_gas_accel(request + 1e-6, pitch), delta=3e-6)

  def test_small_downhill_requests_preserve_wire_accel_and_stable_domains(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request in (-0.05, 0.0, 0.02, 0.15):
      gas = []
      for pitch in (0.0, -0.03, -0.08):
        with self.subTest(request=request, pitch=pitch):
          safety = libsafety_py.libsafety
          config = CP.safetyConfigs[-1]
          self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
          safety.init_tests()
          safety.set_controls_allowed(True)
          controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
          controller.odyssey_pitch.x = pitch
          controller.odyssey_gas_selected = True
          control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., pitch, 0.])
          control.actuators.accel = request
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
          state = structs.CarState(vEgo=22., aEgo=request, canValid=True)
          CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                               odyssey_car_gas=0., odyssey_shift_activity=119, odyssey_target_gear=7,
                               odyssey_user_brake=0., odyssey_computer_braking=False)
          parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
          for frame in range(100):
            now = 1_000_000_000 + frame * 10_000_000
            CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
            output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
            messages = [msg for msg in sends if msg[0] == 0x1DF]
            if not messages:
              continue
            for addr, dat, bus in messages:
              self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))
            parser.update([(now, messages)])
            signals = parser.vl['ACC_CONTROL']
            self.assertAlmostEqual(signals['ACCEL_COMMAND'], request, delta=.01)
            self.assertAlmostEqual(output.accel, request, delta=.01)
            self.assertEqual(signals['BRAKE_REQUEST'], 0)
            if request > 0.0:
              self.assertGreaterEqual(signals['GAS_COMMAND'], 0)
            if request < 0.0 and pitch < 0.0:
              self.assertEqual(signals['GAS_COMMAND'], -30000)
          gas.append(parser.vl['ACC_CONTROL']['GAS_COMMAND'])
      self.assertLess(gas[1], gas[0] - 100)
      self.assertLessEqual(gas[2], gas[1])

  def test_negative_request_uphill_gas_scales_with_grade_without_reversing_request_order(self):
    self.assertEqual(ODYSSEY_ZERO_GAS_GRADE_GAIN, 0.25)
    mild = odyssey_grade_gas_accel(-0.20, 0.02)
    steep = odyssey_grade_gas_accel(-0.20, 0.07)
    self.assertLess(mild, -0.10)
    self.assertGreater(steep, -0.10)
    self.assertGreater(odyssey_grade_gas_accel(-0.10, 0.07), 0.0)
    for pitch in (0.001, 0.02, 0.07, 0.20):
      requests = [ODYSSEY_ROAD_BRAKE_ENTRY + i * 0.001 for i in range(501)]
      mapped = [odyssey_grade_gas_accel(request, pitch) for request in requests]
      self.assertTrue(all(left <= right + 1e-10 for left, right in zip(mapped, mapped[1:], strict=False)))
      self.assertAlmostEqual(odyssey_grade_gas_accel(-0.20 - 1e-6, pitch),
                             odyssey_grade_gas_accel(-0.20 + 1e-6, pitch), delta=5e-6)
      self.assertAlmostEqual(odyssey_grade_gas_accel(-1e-6, pitch),
                             odyssey_grade_gas_accel(0.0, pitch), delta=5e-6)
      self.assertAlmostEqual(odyssey_grade_gas_accel(0.0, pitch),
                             odyssey_grade_gas_accel(1e-6, pitch), delta=5e-6)
    self.assertGreater(odyssey_grade_gas_accel(-0.10, 0.08) - odyssey_grade_gas_accel(-0.20, 0.08), 0.09)
    self.assertLessEqual(odyssey_grade_gas_accel(0.0, 0.50), ODYSSEY_UPHILL_GAS_ACCEL_MAX)
    self.assertEqual(odyssey_grade_weight(0.0), 0.0)
    self.assertTrue(0.0 < odyssey_grade_weight(0.02) < 1.0)
    self.assertEqual(odyssey_grade_weight(0.05), 1.0)

  def test_gas_feedforward_keeps_request_order_with_calibrated_uphill_ceiling(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    params = CarControllerParams(CP)
    for speed, pitch, load_factor in product((8., 10., 14., 22., 24., 30.),
                                             (-.08, 0., .02, .04, .08, .15, .5), ODYSSEY_LOAD_FACTORS):
      with self.subTest(speed=speed, pitch=pitch, load_factor=load_factor):
        commands = []
        for request in (i * .01 for i in range(-30, 201)):
          gas, weight = odyssey_gas_feedforward(request, odyssey_grade_gas_accel(request, pitch), speed, params, load_factor)
          self.assertTrue(0. <= gas <= params.BOSCH_GAS_LOOKUP_V[-1])
          self.assertTrue(0. <= weight <= 1.)
          commands.append(gas)
        self.assertTrue(all(left <= right + 1e-9 for left, right in zip(commands, commands[1:], strict=False)))

  def test_uphill_gas_cap_feedback_weight_tracks_limited_calibrated_grade(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    params = CarControllerParams(CP)

    def mapped(accel):
      return (accel + .2) / 2.2 * 2000.

    for request, pitch in ((-.1, .03), (.95, -.03), (.3, .03)):
      self.assertEqual(odyssey_gas_feedforward(request, odyssey_grade_gas_accel(request, pitch), 19., params)[1], 0.)
    request, pitch = .95, .04
    gas_accel = odyssey_grade_gas_accel(request, pitch)
    gas, weight = odyssey_gas_feedforward(request, gas_accel, 19., params)
    grade = gas_accel - request
    load = min(grade / ODYSSEY_GRADE_RAMP_ACCEL, 1.)
    base = odyssey_low_speed_gas_command(mapped(request), request, 19.)
    uncapped = odyssey_low_speed_gas_command(mapped(gas_accel), request, 19.)
    ceiling = odyssey_low_speed_gas_command(mapped(1.), 1., 19.)
    self.assertAlmostEqual(gas, ceiling)
    self.assertAlmostEqual(weight, (uncapped - ceiling) / (uncapped - base) * load * load * (3. - 2. * load))
    high_input = odyssey_grade_gas_accel(1.2, pitch)
    high_gas, high_weight = odyssey_gas_feedforward(1.2, high_input, 19., params)
    self.assertAlmostEqual(high_gas, odyssey_low_speed_gas_command(mapped(1.2), 1.2, 19.))
    self.assertAlmostEqual(high_weight, load * load * (3. - 2. * load))

  def test_uphill_gas_cap_feedback_fades_with_vanishing_grade(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    params = CarControllerParams(CP)
    for pitch, lower, upper in ((0., 0., 0.), (1e-5, 0., 1e-3), (.08, .9, 1.)):
      weight = odyssey_gas_feedforward(1.2, odyssey_grade_gas_accel(1.2, pitch), 19., params)[1]
      self.assertGreaterEqual(weight, lower)
      self.assertLessEqual(weight, upper)

  def test_gas_response_cap_weight_increases_only_bounded_feedback(self):
    self.assertGreater(ODYSSEY_RESPONSE_CAPPED_COUNTS_PER_ACCEL, ODYSSEY_RESPONSE_COUNTS_PER_ACCEL)
    baseline, capped = OdysseyGasResponse(), OdysseyGasResponse()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 20):
      base = baseline.update(0.95, 0.65, 0.04, 19.0, 1, True)
      previous = capped.correction
      extra = capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 1.0)
      self.assertLessEqual(abs(extra - previous), 10.0)
    self.assertGreater(base, 60.0)
    self.assertLess(base, extra)
    self.assertEqual(extra, ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertLessEqual(capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 0.0), extra)
    self.assertEqual(capped.update(0.95, 0.65, 0.04, 19.0, 1, False, 1.0), 0.0)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      self.assertEqual(capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 1.0), 0.0)
    self.assertGreater(capped.update(0.95, 0.65, 0.04, 19.0, 1, True, 1.0), 0.0)

  def test_gas_response_waits_for_vehicle_delay_before_adding_gas(self):
    response = OdysseyGasResponse()

    def update(request=0.0, aego=-0.3, pitch=0.06, gear=1, eligible=True):
      return response.update(request, aego, pitch, 20.0, gear, eligible)

    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      self.assertEqual(update(), 0.0)
    self.assertGreater(update(), 0.0)
    for _ in range(100):
      update()
    self.assertGreater(response.correction, 0.0)
    self.assertLessEqual(response.correction, ODYSSEY_RESPONSE_MAX_COUNTS)

    # A stronger deceleration request must unwind the old positive correction.
    previous = response.correction
    self.assertLess(update(request=-0.2), previous)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES * 2):
      update(request=-0.2, aego=0.2)
    self.assertLess(response.correction, 0.0)
    self.assertEqual(update(eligible=False), 0.0)
    self.assertEqual(response.correction, 0.0)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      self.assertEqual(update(gear=2), 0.0)
    self.assertGreater(update(gear=2), 0.0)

  def test_launch_gas_response_waits_for_fresh_released_brakes(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for econ, aego, feedback in product((False, True), (0., 1.), ('computer', 'pressure', 'stale', 'missing')):
      with self.subTest(econ=econ, aego=aego, feedback=feedback):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        state = structs.CarState(vEgo=0., aEgo=aego, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False,
                             acc_hud={}, lkas_hud={}, stock_brake={}, odyssey_car_gas=0.,
                             odyssey_engine_torque_ts_nanos=0, odyssey_target_gear=1, odyssey_shift_activity=30,
                             odyssey_target_gear_ts_nanos=0, odyssey_user_brake=0.,
                             odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = .3
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(132):
          now = 1_000_000_000 + frame * 10_000_000
          holding = frame < 80
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = now
          CS.odyssey_computer_braking_ts_nanos = (0 if feedback == 'missing' else now - 100_000_000) if holding else now
          if feedback in ('computer', 'pressure'):
            CS.odyssey_computer_braking_ts_nanos = now
          CS.odyssey_computer_braking = holding and feedback == 'computer'
          CS.odyssey_user_brake = .03125 if holding and feedback == 'pressure' else 0.
          if not holding:
            state.aEgo = 0.
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, sends)])
          wire = parser.vl['ACC_CONTROL']
          self.assertAlmostEqual(wire['ACCEL_COMMAND'], .3, delta=.005)
          self.assertEqual(wire['BRAKE_REQUEST'], 0)
          self.assertEqual(wire['STANDSTILL'], 0)
          self.assertEqual(wire['STANDSTILL_RELEASE'], 1)
          self.assertGreater(wire['GAS_COMMAND'], 0.)
          response = controller.odyssey_gas_response
          if holding:
            self.assertFalse(response.requests)
            self.assertEqual(response.integral, 0.)
            self.assertEqual(response.correction, 0.)
            self.assertAlmostEqual(output.gas, 2000. * (.3 + .2) / 2.2, delta=1.)
          elif frame < 130:
            self.assertEqual(response.correction, 0.)
        self.assertGreater(response.correction, 0.)

  def test_gas_response_reduces_startup_overacceleration_without_adding_gas(self):
    response = OdysseyGasResponse()
    for _ in range(15):
      self.assertEqual(response.update(0.02, -0.15, -0.03, 30.0, 1, True), 0.0)
    for _ in range(5):
      previous = response.correction
      correction = response.update(-0.02, 0.15, -0.03, 30.0, 1, True)
      self.assertLess(correction, 0.0)
      self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
      self.assertLessEqual(abs(correction), ODYSSEY_RESPONSE_MAX_COUNTS)
    previous = response.correction
    self.assertLess(response.update(0.04, 1.0, -0.03, 30.0, 1, True), previous)
    previous = response.correction
    self.assertGreater(response.update(0.04, 0.0, -0.03, 30.0, 1, True), previous)
    self.assertEqual(response.update(-0.02, 0.15, -0.03, 30.0, 1, False), 0.0)
    self.assertFalse(response.requests)
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES - 1):
      previous = response.correction
      correction = response.update(0.02, 1.0, -0.03, 30.0, 1, True, cap_weight=1.0)
      self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
      self.assertLessEqual(abs(correction), ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertEqual(correction, -ODYSSEY_RESPONSE_MAX_COUNTS)

  def test_gas_response_corrects_excess_while_startup_request_rises(self):
    response = OdysseyGasResponse()
    self.assertEqual(response.update(.1, .1, .02, 20.0, 7, True), 0.0)
    self.assertLess(response.update(.11, .4, .02, 20.0, 7, True), 0.0)

  def test_gas_response_can_remove_available_gas_during_startup(self):
    for available in (60., 600.):
      with self.subTest(available=available):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
          previous = response.correction
          correction = response.update(.2, 1.2, 0., 20., 6, True, limits=(-available, 700.))
          self.assertGreaterEqual(correction, -available)
          self.assertLessEqual(correction, 0.)
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertEqual(response.integral, 0.)
        if available > ODYSSEY_RESPONSE_MAX_COUNTS:
          self.assertLess(correction, -ODYSSEY_RESPONSE_MAX_COUNTS)
        else:
          self.assertEqual(correction, -available)
        for _ in range(100):
          previous = response.correction
          correction = response.update(.2, 1.2, 0., 20., 6, True, limits=(-available, 700.))
          self.assertGreaterEqual(correction, -available)
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertGreaterEqual(response.integral, -available)
          self.assertLessEqual(response.integral, ODYSSEY_RESPONSE_MAX_COUNTS)
        issued = correction
        for _ in range(math.ceil(available / ODYSSEY_RESPONSE_SLEW_COUNTS) + ODYSSEY_RESPONSE_DELAY_FRAMES):
          previous = response.correction
          correction = response.update(.2, .2, 0., 20., 7, True, limits=(-available, 700.))
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertLessEqual(correction, 0.)
        self.assertEqual(correction, issued)
        for _ in range(100):
          correction = response.update(.2, -.8, 0., 20., 7, True, limits=(-available, 700.))
          self.assertLessEqual(correction, 700.)
          self.assertLessEqual(response.integral, 700.)

  def test_gas_feedback_reaches_zero_without_changing_acceleration(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    controller.odyssey_pitch.x = -.03
    state = structs.CarState(vEgo=20., aEgo=.9, canValid=True)
    CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=.1, odyssey_engine_torque_ts_nanos=0, odyssey_target_gear=7,
                         odyssey_shift_activity=119, odyssey_target_gear_ts_nanos=0,
                         odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., -.03, 0.])
    control.actuators.accel = .1
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    for frame in range(120):
      now = 1_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
      previous = controller.odyssey_gas_response.correction
      output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      parser.update([(now, sends)])
      self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], .1, delta=.005)
      self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
      self.assertAlmostEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], output.gas, delta=1.)
      self.assertGreaterEqual(output.gas, 0.)
      response = controller.odyssey_gas_response
      self.assertLessEqual(abs(response.correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
      self.assertLessEqual(abs(response.integral), ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertLess(response.correction, -ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertEqual(output.gas, 0.)

  def test_gas_response_uses_latest_tracking_error_with_delayed_feedback(self):
    for request, aego, direction in ((.7, .9, -1), (.3, .1, 1)):
      with self.subTest(request=request):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 1):
          self.assertEqual(response.update(.5, .5, .02, 20.0, 7, True), 0.0)
        self.assertGreater(direction * response.update(request, aego, .02, 20.0, 7, True), 0.0)

    for request, aego in ((.55, .525), (.45, .475)):
      with self.subTest(request=request):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 1):
          response.update(.5, .5, .02, 20.0, 7, True)
        self.assertEqual(response.update(request, aego, .02, 20.0, 7, True), 0.0)

    for aego, request in ((.2, .21), (.8, .79)):
      with self.subTest(aego=aego):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 30):
          response.update(.5, aego, .02, 20.0, 7, True)
        retained = response.integral
        for _ in range(15):
          previous = response.correction
          correction = response.update(request, aego, .02, 20.0, 7, True)
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
        # A smaller current error limits the proportional term, while the offset remains bounded.
        self.assertLessEqual(abs(response.integral - retained), 200.0 * abs(request - aego) * .3 + 1e-6)
        self.assertAlmostEqual(correction - response.integral, 200.0 * (request - aego))

  def test_gas_response_retains_correction_at_requested_acceleration(self):
    for direction in (-1, 1):
      with self.subTest(direction=direction):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 100):
          response.update(.2, .2 - direction * .03, 0., 20., 6, True)
        previous = response.correction
        for _ in range(50):
          correction = response.update(.2, .2, 0., 20., 6, True)
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          previous = correction
        self.assertGreater(direction * correction, 10.)
        self.assertLess(abs(correction), ODYSSEY_RESPONSE_MAX_COUNTS)

  def test_gas_response_converges_after_request_change_with_available_headroom(self):
    # Illustrative delayed plant: a convergence regression, not an identified Honda gas gain.
    for gain in (.001, .003):
      for bias in (-.3, -.06, .06, .3):
        for cap in (0., 1.):
          with self.subTest(gain=gain, bias=bias, cap=cap):
            response = OdysseyGasResponse()
            delay = deque([.2 + bias] * 36)
            achieved = .2 + bias
            previous_command = .2
            errors = []
            for frame in range(2000):
              request = .2 if frame < 1000 else .1
              correction = response.update(request, achieved, 0., 20., 6, True, cap_weight=cap, limits=(-600., 600.))
              command = request + gain * correction
              if frame == 1000:
                self.assertLess(command, previous_command)
              previous_command = command
              delay.append(command + bias)
              achieved += (delay.popleft() - achieved) * DT_CTRL * 2 / (.15 + DT_CTRL * 2)
              if frame >= 1900:
                errors.append(abs(request - achieved))
            self.assertLess(sum(errors) / len(errors), .003)

  def test_gas_response_retained_reduction_follows_available_headroom(self):
    response = OdysseyGasResponse()
    for _ in range(300):
      response.update(.2, .4, 0., 20., 6, True, limits=(-600., 100.))
    self.assertLess(response.integral, -ODYSSEY_RESPONSE_MAX_COUNTS)
    for _ in range(30):
      previous = response.correction
      correction = response.update(.2, .2, 0., 20., 6, True, limits=(-50., 100.))
      self.assertGreaterEqual(response.integral, -50.)
      self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
    self.assertEqual(correction, -50.)
    self.assertEqual(response.update(.2, .2, 0., 20., 6, False), 0.)
    self.assertEqual(response.integral, 0.)

  def test_gas_response_saturation_unwinds_without_storing_excess(self):
    for direction in (-1, 1):
      with self.subTest(direction=direction):
        response = OdysseyGasResponse()
        for _ in range(300):
          response.update(.2, .2 - direction * 2., 0., 20., 6, True)
        self.assertEqual(response.correction, direction * ODYSSEY_RESPONSE_MAX_COUNTS)
        self.assertEqual(response.integral, 0.)
        for _ in range(100):
          previous = response.correction
          response.update(.2, .2 + direction * .3, 0., 20., 6, True)
          self.assertLessEqual(abs(response.correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertLessEqual(abs(response.correction), ODYSSEY_RESPONSE_MAX_COUNTS)
        self.assertLess(direction * response.correction, 0.)

  def test_gas_response_does_not_store_unavailable_correction(self):
    for direction, limits in ((-1, (-5., 100.)), (1, (-100., 5.))):
      with self.subTest(direction=direction):
        response = OdysseyGasResponse()
        for _ in range(300):
          response.update(.2, .2 - direction * .03, 0., 20., 6, True, limits=limits)
        self.assertEqual(response.correction, direction * 5.)
        self.assertEqual(response.integral, 0.)
        self.assertEqual(response.update(.2, .2, 0., 20., 6, True, limits=limits), 0.)
        for _ in range(50):
          response.update(.2, .2 - direction * .05, 0., 20., 6, True)
        self.assertGreater(direction * response.integral, 5.)
        for _ in range(250):
          response.update(.2, .2 + direction * .01, 0., 20., 6, True, limits=limits)
        self.assertLess(direction * response.integral, 5.)
        self.assertLess(direction * response.correction, 0.)
        invalid_limits = (math.nan, 100.) if direction < 0 else (-100., math.nan)
        self.assertEqual(response.update(.2, .2, 0., 20., 6, True, limits=invalid_limits), 0.)
        self.assertEqual(response.integral, 0.)

  def test_gas_response_preserves_issued_correction_with_available_headroom(self):
    for upper in (150., 600.):
      with self.subTest(upper=upper):
        response = OdysseyGasResponse()
        for _ in range(300):
          previous = response.correction
          correction = response.update(.2, -.5, .04, 20., 6, True, limits=(-600., upper))
          self.assertLessEqual(correction, upper)
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertLessEqual(response.integral, upper)
        self.assertGreater(correction, ODYSSEY_RESPONSE_MAX_COUNTS)
        issued = correction
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
          correction = response.update(.2, -.5, .04, 20., 7, True, limits=(-600., upper))
          self.assertEqual(correction, issued)
          self.assertLessEqual(response.integral, upper)
        for _ in range(100):
          previous = response.correction
          correction = response.update(.2, .2, .04, 20., 7, True, limits=(-600., upper))
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
        self.assertEqual(correction, issued)

  def test_gas_response_warmup_reduction_respects_headroom(self):
    for lower, achieved in product((-5., -50.), (0., .2, .4)):
      with self.subTest(lower=lower, achieved=achieved):
        response = OdysseyGasResponse()
        for _ in range(300):
          response.update(.2, .4, 0., 20., 6, True, limits=(-600., 600.))
        self.assertLess(response.correction, -50.)
        issued = response.correction
        response.clear_observer()
        correction = response.update(.2, achieved, 0., 20., 6, True, limits=(lower, 600.))
        self.assertLessEqual(abs(correction - issued), ODYSSEY_RESPONSE_SLEW_COUNTS)
        self.assertLessEqual(correction, 0.)
        self.assertGreaterEqual(response.integral, lower)
        self.assertLessEqual(response.integral, 0.)
        if (achieved < .2 or achieved > .2) and lower == -5.:
          self.assertEqual(response.integral, 0.)
        else:
          self.assertLess(response.integral, 0.)
        self.assertFalse(response.error.initialized)
        self.assertEqual(len(response.requests), 1)

  def test_shift_sample_does_not_erase_gas_reduction(self):
    for change in ('gear', 'activity', 'clear'):
      with self.subTest(change=change):
        response = OdysseyGasResponse()
        for _ in range(300):
          response.update(.2, .4, 0., 20., 6, True, shift_activity=102, limits=(-600., 600.))
        issued = response.correction
        self.assertLess(issued, -50.)
        if change == 'clear':
          response.clear_observer()
        gear = 7 if change == 'gear' else 6
        activity = 110 if change == 'activity' else 102
        retained = issued + .02 * ODYSSEY_RESPONSE_COUNTS_PER_ACCEL
        for frame in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
          achieved = .18 if frame == 0 else .1 if frame == 1 else .2
          if frame == 1:
            retained += ODYSSEY_RESPONSE_SLEW_COUNTS
          correction = response.update(.2, achieved, 0., 20., gear, True,
                                       shift_activity=activity, limits=(-600., 600.))
          self.assertAlmostEqual(correction, retained)
          self.assertFalse(response.error.initialized)
        # Fresh sustained under-response still restores gas after the observer is ready.
        for _ in range(100):
          previous = response.correction
          correction = response.update(.2, .1, 0., 20., gear, True,
                                       shift_activity=activity, limits=(-600., 600.))
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
        self.assertGreater(correction, retained)

  def test_gas_response_reverse_error_respects_integral_headroom(self):
    for lower in (-5., -50., -600.):
      with self.subTest(lower=lower):
        response = OdysseyGasResponse()
        response.correction = 20.
        response.update(.2, .7, 0., 20., 6, True, limits=(lower, 600.))
        self.assertEqual(response.integral, 0.)
        self.assertLessEqual(abs(response.correction - 20.), ODYSSEY_RESPONSE_SLEW_COUNTS)

  def test_gas_feedback_preserves_headroom_after_a_request_change(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request, achieved, next_request, pitch in ((-.19, .11, -.17, 0.), (2., 1.7, 1.8, 0.),
                                                  (1.95, 1.35, 1.8, 0.), (.6, .9, .8, -.115)):
      with self.subTest(request=request):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.odyssey_pitch.x = pitch
        state = structs.CarState(vEgo=20., canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=.1, odyssey_engine_torque_ts_nanos=0, odyssey_target_gear=7,
                             odyssey_shift_activity=0, odyssey_target_gear_ts_nanos=0,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., pitch, 0.])
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(320):
          control.actuators.accel = .2 if frame < 2 else request if frame < 300 else next_request
          state.aEgo = .2 if frame < 2 else achieved if frame < 300 else next_request
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, sends)])
          signals = parser.vl['ACC_CONTROL']
          self.assertEqual(signals['BRAKE_REQUEST'], 0)
          self.assertAlmostEqual(signals['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
          self.assertAlmostEqual(signals['GAS_COMMAND'], output.gas, delta=1.)
          if frame == 298:
            saturated = controller.params.BOSCH_GAS_LOOKUP_V[-1] if achieved < request else 0.
            self.assertAlmostEqual(output.gas, saturated, delta=1e-4)
        self.assertEqual(controller.odyssey_gas_response.integral, 0.)
        gas_request = odyssey_grade_gas_accel(next_request, controller.odyssey_pitch.x)
        mapped, _ = odyssey_gas_feedforward(next_request, gas_request, state.vEgo, controller.params)
        self.assertAlmostEqual(output.gas, mapped, delta=1e-4)

  def test_gas_response_drops_retained_offset_when_feedback_is_invalid(self):
    for change in ('inactive', 'invalid'):
      with self.subTest(change=change):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 100):
          response.update(.2, 0., .04, 20., 6, True, shift_activity=102)
        self.assertGreater(response.integral, 0.)
        for _ in range(10):
          response.update(.2, .2, .04, 20., 6, change != 'inactive',
                          shift_activity=102,
                          cap_weight=math.nan if change == 'invalid' else 0.)
        self.assertEqual(response.integral, 0.)

  def test_gas_response_does_not_reintroduce_error_after_observer_warmup(self):
    # Keep the illustrative plant unchanged to isolate the observer's state transition.
    for gain, bias, change in product((.001, .003), (-.3, -.15, -.06, .06, .15, .3), ('gear', 'activity', 'grade', 'clear')):
      with self.subTest(gain=gain, bias=bias, change=change):
        response = OdysseyGasResponse()
        achieved = .2 - bias
        delay = deque([achieved] * 25)
        errors = []
        for frame in range(2200):
          changed = frame >= 2000
          if frame == 2000 and change == 'clear':
            response.clear_observer()
          correction = response.update(.2, achieved, .02 if changed and change == 'grade' else .04,
                                       20., 7 if changed and change == 'gear' else 6, True,
                                       shift_activity=110 if changed and change == 'activity' else 102,
                                       limits=(-600., 600.))
          delay.append(.2 - bias + gain * correction)
          achieved += (delay.popleft() - achieved) * .02 / (.15 + .02)
          if changed:
            errors.append(abs(.2 - achieved))
        self.assertLess(max(errors), .003)

  def test_gas_response_preserves_issued_correction_through_observer_warmup(self):
    for change in ('gear', 'activity', 'grade'):
      with self.subTest(change=change):
        response = OdysseyGasResponse()
        for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES + 50):
          response.update(.2, .1, .04, 20., 6, True, shift_activity=102)
        issued = response.correction
        self.assertGreater(issued, 20.)
        self.assertLess(issued, ODYSSEY_RESPONSE_MAX_COUNTS)
        gear = 7 if change == 'gear' else 6
        activity = 110 if change == 'activity' else 102
        pitch = .02 if change == 'grade' else .04
        for _ in range(4):
          correction = response.update(.2, .1, pitch, 20., gear, True, shift_activity=activity)
          self.assertEqual(correction, issued)
          self.assertEqual(response.integral, correction)
        self.assertLess(response.update(.2, .1, pitch, 20., gear, True, shift_activity=activity, limits=(-100., 0.)), issued)
        self.assertEqual(response.integral, 0.)
        # Holding an issued correction cannot add gas or exceed current headroom.
        for _ in range(5):
          previous = response.correction
          correction = response.update(.2, .2, pitch, 20., gear, True, shift_activity=activity, limits=(-100., 15.))
          self.assertLessEqual(correction, previous)
          self.assertLessEqual(previous - correction, ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertLessEqual(response.integral, 15.)
        self.assertEqual(correction, 15.)
        # Current excess response still removes gas before the new observer is ready.
        for _ in range(5):
          previous = response.correction
          correction = response.update(.2, .4, pitch, 20., gear, True, shift_activity=activity)
          self.assertLessEqual(correction, previous)
          self.assertLessEqual(previous - correction, ODYSSEY_RESPONSE_SLEW_COUNTS)
        self.assertLess(correction, 0.)
        self.assertLess(len(response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES)
        self.assertEqual(response.integral, 0.)
        self.assertEqual(response.update(.2, .1, pitch, 20., gear, False, shift_activity=activity), 0.)
        self.assertFalse(response.requests)

  def test_gas_response_corrects_residual_during_falling_grade(self):
    for achieved in (-.3, .3):
      with self.subTest(achieved=achieved):
        response = OdysseyGasResponse()
        corrections = []
        for frame in range(ODYSSEY_RESPONSE_DELAY_FRAMES * 4):
          previous = response.correction
          correction = response.update(0., achieved, .06 - frame * .0005, 20., 6, True)
          self.assertLessEqual(abs(correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          corrections.append(correction)
        if achieved < 0.:
          self.assertTrue(all(c > 0. for c in corrections[ODYSSEY_RESPONSE_DELAY_FRAMES:]))
        else:
          self.assertLess(corrections[-1], -ODYSSEY_RESPONSE_COUNTS_PER_ACCEL * achieved)

  def test_gas_response_does_not_reintroduce_excess_when_grade_falls(self):
    # The illustrative plant keeps its required correction as the grade changes.
    for gain in (.001, .003):
      with self.subTest(gain=gain):
        response = OdysseyGasResponse()
        achieved = .26
        delay = deque([achieved] * ODYSSEY_RESPONSE_DELAY_FRAMES)
        errors = []
        for frame in range(2200):
          pitch = .04 - max(frame - 2000, 0) * .0005
          correction = response.update(.2, achieved, pitch, 20., 6, True)
          delay.append(.26 + gain * correction)
          achieved += (delay.popleft() - achieved) * .02 / (.15 + .02)
          if frame >= 2000:
            errors.append(abs(.2 - achieved))
        self.assertLess(max(errors), .003)

  def test_gas_response_rejects_stale_lead_request(self):
    response = OdysseyGasResponse()
    for _ in range(ODYSSEY_RESPONSE_DELAY_FRAMES):
      assert response.update(0.3, -0.3, 0.06, 20.0, 1, True) == 0.0
    assert response.update(-0.3, -0.3, 0.06, 20.0, 1, True) == 0.0

  def test_gas_response_does_not_reintroduce_excess_after_grade_transition(self):
    # Illustrative delayed propulsion with correct grade feedforward and a fixed gas-map bias.
    # A transient uphill shortfall does not mean the learned reduction became invalid.
    def run(observe_grade):
      response = OdysseyGasResponse()
      grade = 9.81 * math.sin(.04)
      propulsion = .2 + grade
      achieved = .5
      delay = deque([propulsion] * 25)
      errors = []
      for frame in range(4000):
        pitch = .04 + .04 * min(max(frame - 2000, 0) * .02, 4.)
        grade = 9.81 * math.sin(pitch)
        correction = response.update(.2, achieved, pitch if observe_grade else .04, 20., 6, True,
                                     limits=(-600., 600.))
        delay.append(.2 + grade + .001 * correction)
        propulsion += (delay.popleft() - propulsion) * .02 / (.15 + .02)
        achieved = propulsion + .3 - grade
        if frame >= 2200:
          errors.append(achieved - .2)
      return max(errors), sum(abs(e) for e in errors[-50:]) / 50

    reference_excess, reference_settled = run(False)
    excess, settled = run(True)
    self.assertLessEqual(excess, reference_excess + .003)
    self.assertLess(reference_settled, .003)
    self.assertLess(settled, .003)

  def test_gas_feedback_uses_fresh_received_target_gear_through_drive_shifts(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    packer = CANPacker(dbc)
    for speed, blocked in product((0., 2., 20.), (None, 'last_fresh', 'missing', 'stale', 'future', 'invalid_can', 'neutral', 'reverse')):
      with self.subTest(speed=speed, blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        CS = CarState(CP, _sp(CP))
        parsers = CS.get_can_parsers(CP, _sp(CP))
        output_parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = .2
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        for frame in range(82):
          now = 1_000_000_000 + frame * 10_000_000
          target = 6 if frame < 80 else 7
          stamp = now - 5_000_000
          if frame >= 80:
            target = 0 if blocked == 'neutral' else 13 if blocked == 'reverse' else target
            stamp = {'last_fresh': now - 59_999_999, 'stale': now - 60_000_000, 'future': now + 1}.get(blocked, stamp)
          parsers[Bus.pt].update([(stamp, [packer.make_can_msg('GEARBOX_AUTO', 1,
                                     {'TRANS_TARGET_GEAR': target, 'GEAR_SHIFTER': 4}),
                                         packer.make_can_msg('VSA_STATUS', 1, {})])])
          state, _ = CS.update(parsers)
          state.vEgo = speed
          state.aEgo = 0.
          state.canValid = not (frame >= 80 and blocked == 'invalid_can')
          CS.out = state
          if frame >= 80 and blocked == 'missing':
            CS.odyssey_target_gear_ts_nanos = 0
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          output_parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          self.assertEqual(str(state.gearShifter), 'drive')
          self.assertAlmostEqual(output_parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], .2, delta=.01)
          self.assertEqual(output_parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
          self.assertGreater(output_parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
          if frame == 78:
            self.assertGreater(controller.odyssey_gas_response.correction, 0)
            self.assertEqual(len(controller.odyssey_gas_response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
          if frame >= 80:
            self.assertEqual(len(controller.odyssey_gas_response.requests), 1 if blocked in (None, 'last_fresh') else 0)
            if blocked not in (None, 'last_fresh'):
              self.assertEqual(controller.odyssey_gas_response.correction, 0)

  def test_gas_feedback_does_not_compare_across_received_transmission_activity(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for speed, changed, achieved in product((0., 2., 20.), (False, True), (0., -.5)):
      with self.subTest(speed=speed, changed=changed, achieved=achieved):
        packer = CANPacker(dbc)
        CS = CarState(CP, _sp(CP))
        parsers = CS.get_can_parsers(CP, _sp(CP))
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        wire = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = .2
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        for frame in range(134):
          now = 1_000_000_000 + frame * 10_000_000
          activity = 110 if changed and frame >= 80 else 102
          parsers[Bus.pt].update([(now - 5_000_000, [packer.make_can_msg('GEARBOX_AUTO', 1,
                                     {'TRANS_TARGET_GEAR': 6, 'TRANS_SHIFT_ACTIVITY': activity, 'GEAR_SHIFTER': 4}),
                                                   packer.make_can_msg('VSA_STATUS', 1, {})])])
          state, _ = CS.update(parsers)
          state.vEgo, state.aEgo, state.canValid = speed, achieved, True
          CS.out = state
          if frame > 0:
            self.assertEqual(CS.odyssey_target_gear, 6)
            self.assertEqual(CS.odyssey_shift_activity, activity)
            self.assertEqual(CS.odyssey_target_gear_ts_nanos, now - 5_000_000)
          previous = controller.odyssey_gas_response.correction
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          if frame % 2 or frame == 0:
            continue
          wire.update([(now, sends)])
          self.assertAlmostEqual(wire.vl['ACC_CONTROL']['ACCEL_COMMAND'], .2, delta=.01)
          self.assertEqual(wire.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
          self.assertGreater(wire.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
          response = controller.odyssey_gas_response
          self.assertLessEqual(abs(response.correction - previous), ODYSSEY_RESPONSE_SLEW_COUNTS)
          if frame == 78:
            self.assertGreater(response.correction, ODYSSEY_RESPONSE_MAX_COUNTS if achieved < 0. else 40.)
          if frame == 80:
            self.assertEqual(len(response.requests), 1 if changed else ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
            if changed:
              self.assertEqual(response.correction, previous)
              self.assertEqual(response.integral, response.correction)
            else:
              self.assertGreater(response.correction, previous)
          if frame == 128 and changed:
            self.assertEqual(len(response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES)
            self.assertGreater(response.correction, 40.)
            self.assertEqual(response.integral, response.correction)
          if frame == 130:
            self.assertGreater(response.correction, 0.)
            if changed:
              self.assertGreaterEqual(response.correction, previous)

  def test_odyssey_low_speed_feedback_is_bounded_and_survives_speed_entry(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]

    def run(request, achieved, speed):
      controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
      parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
      control = structs.CarControl()
      control.enabled = True
      control.longActive = True
      control.actuators.accel = request
      control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
      control.orientationNED = [0.0, 0.0, 0.0]
      state = structs.CarState()
      state.vEgo = speed
      state.aEgo = achieved
      state.canValid = True
      CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                           odyssey_car_gas=math.nan,
                           odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=3,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

      def step(frame):
        CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = frame * 10_000_000
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, frame * 10_000_000)
        messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
        if messages:
          parser.update([(frame * 10_000_000, messages)])
          self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], request, delta=0.01)
          self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        return parser.vl['ACC_CONTROL']['GAS_COMMAND']

      values = [step(frame) for frame in range(100)]
      return controller, state, step, values

    # Gas response follows the tracking error through launch and the domain speed boundary.
    for speed in (0., 2., 4.9, 5., 7.):
      with self.subTest(speed=speed):
        _, _, _, baseline = run(.75, .75, speed)
        for achieved in (.45, 1.05):
          controller, state, step, values = run(.75, achieved, speed)
          response = controller.odyssey_gas_response
          self.assertEqual(len(response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
          if achieved < .75:
            self.assertGreater(values[-1], baseline[-1] + 40)
            self.assertEqual(values[:52], baseline[:52])
          else:
            self.assertLess(values[-1], baseline[-1] - 40)
          integral = response.integral
          state.vEgo = 5.
          self.assertLessEqual(abs(step(100) - values[-1]), ODYSSEY_RESPONSE_SLEW_COUNTS)
          self.assertEqual(len(response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
          self.assertLess(abs(response.integral - integral), 2.)

    moderate, state, step, values = run(0.75, 1.05, 7.0)
    bp, gas_values = moderate.params.BOSCH_GAS_LOOKUP_BP, moderate.params.BOSCH_GAS_LOOKUP_V
    mapped = gas_values[0] + (.75 - bp[0]) / (bp[-1] - bp[0]) * (gas_values[-1] - gas_values[0])
    self.assertGreaterEqual(moderate.odyssey_gas_response.correction, -odyssey_low_speed_gas_command(mapped, .75, 7.0))
    state.vEgo = 8.0
    self.assertLessEqual(abs(step(100) - values[-1]), 10)

    high, state, step, values = run(.1, -.6, 20.0)
    self.assertGreater(high.odyssey_gas_response.correction, ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertLessEqual(high.odyssey_gas_response.integral, ODYSSEY_RESPONSE_MAX_COUNTS)
    self.assertLess(max(values), CarControllerParams(CP).BOSCH_GAS_LOOKUP_V[-1])
    state.aEgo = .1
    for frame in range(100, 200):
      step(frame)
    self.assertLessEqual(high.odyssey_gas_response.correction, ODYSSEY_RESPONSE_MAX_COUNTS)

    for request in (1.2, 1.6, 1.95, 2.0):
      with self.subTest(request=request):
        _, _, _, uncorrected = run(request, request, 7.0)
        high, state, step, values = run(request, request + .3, 7.0)
        self.assertLess(high.odyssey_gas_response.correction, -40.)
        self.assertLess(values[-1], uncorrected[-1] - 40)
        state.vEgo = 8.0
        self.assertLessEqual(abs(step(100) - values[-1]), ODYSSEY_RESPONSE_SLEW_COUNTS)

        high, _, step, values = run(request, request - .3, 7.0)
        values.extend(step(frame) for frame in range(100, 300))
        bp, gas_values = high.params.BOSCH_GAS_LOOKUP_BP, high.params.BOSCH_GAS_LOOKUP_V
        ceiling = gas_values[-1]
        mapped = gas_values[0] + (request - bp[0]) / (bp[-1] - bp[0]) * (ceiling - gas_values[0])
        self.assertLessEqual(max(values), ceiling)
        self.assertLessEqual(high.odyssey_gas_response.correction, ceiling - mapped + 1e-6)
        self.assertLessEqual(high.odyssey_gas_response.integral, ceiling - mapped + 1e-6)
        if request in (1.2, 1.6):
          self.assertGreater(high.odyssey_gas_response.correction, ODYSSEY_RESPONSE_MAX_COUNTS)
        if request == 2.0:
          self.assertEqual(values[-1], ceiling)
          self.assertEqual(high.odyssey_gas_response.integral, 0.)
        else:
          self.assertGreater(high.odyssey_gas_response.correction, 0.)

  def test_negative_gas_bridge_only_enters_from_road_speed_coast(self):
    self.assertEqual(odyssey_command_domains(-0.10, 20.0), (True, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True, bridge_active=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.11, 20.0, previous_gas=True), (True, False))
    self.assertEqual(odyssey_command_domains(-0.05, 20.0, previous_brake=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.05, 4.0), (False, True))

  def test_brake_domain_follows_acceleration_relative_to_coast(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    controller.odyssey_brake_selected = True
    controller.odyssey_pitch.x = -0.03
    controller.odyssey_coast_response.drag[7] = 0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.03)
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., -0.03, 0.])
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState(vEgo=20., aEgo=0., canValid=True)
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0., odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0,
                         odyssey_target_gear=7, odyssey_target_gear_ts_nanos=0,
                         odyssey_user_brake=0., odyssey_computer_braking=True, odyssey_computer_braking_ts_nanos=0)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    for request in (0., 0.05, 0.1, 0.4):
      control.actuators.accel = request
      for _ in range(2):
        now = 1_000_000_000 + controller.frame * 10_000_000
        CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
        output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
      signals = parser.vl['ACC_CONTROL']
      self.assertAlmostEqual(signals['ACCEL_COMMAND'], request, delta=.01)
      self.assertAlmostEqual(output.accel, request, delta=.01)
      self.assertEqual(signals['BRAKE_REQUEST'], request < 0.3)
      if request < 0.3:
        self.assertEqual(signals['GAS_COMMAND'], -30000)
      else:
        self.assertEqual(signals['GAS_COMMAND'], -30000)
        CS.odyssey_computer_braking = False
        for _ in range(2):
          now += 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        self.assertGreater(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)

  def test_passive_coast_forecast_is_prior_only_and_resets_on_ineligible_state(self):
    response = OdysseyCoastResponse()
    for frame in range(0, 50, 2):
      self.assertIsNone(response.update(frame, 7, 0.0, 9.0, True, True))
    self.assertFalse(response.warmup)
    for frame, accel in zip(range(50, 60, 2), [0.25, 0.24, 3.0, 0.26, 0.27], strict=True):
      self.assertIsNone(response.update(frame, 7, 0.0, accel, True, True))
    self.assertAlmostEqual(response.update(60, 7, 0.0, 0.28, True, True), 0.26)
    self.assertEqual(len(response.warmup[7]), 5)
    for frame in range(62, 80, 2):
      self.assertAlmostEqual(response.update(frame, 7, 0.0, 0.26 if frame == 70 else 9.0, True, True), 0.27)
    self.assertAlmostEqual(response.update(80, 7, 0.0, 0.25, True, True), 0.27)
    self.assertAlmostEqual(response.update(82, 7, 0.0, -3.0, False, True), 0.26)
    samples = list(response.warmup[7])
    self.assertAlmostEqual(response.update(90, 7, 0.0, -3.0, False, True), 0.26)
    self.assertEqual(list(response.warmup[7]), samples)
    for frame in range(92, 140, 2):
      self.assertIsNone(response.update(frame, 8, 0.0, 0.0, True, True))
    # A known gear keeps its prior forecast while new coast observations settle.
    for frame in range(140, 190, 2):
      self.assertAlmostEqual(response.update(frame, 7, 0.0, 9.0, True, True), 0.26)
    self.assertNotIn(7, response.warmup)
    self.assertIsNone(response.update(190, 7, 0.0, 0.0, False, False))
    self.assertFalse(response.drag)
    self.assertFalse(response.warmup)

  def test_nonnegative_brake_hold_does_not_learn_its_response_as_coast(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request in (0.0, 0.03, 0.08):
      with self.subTest(request=request):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.odyssey_pitch.x = -0.02
        for frame in range(0, 100, 2):
          controller.odyssey_coast_response.update(frame, 7, -0.02, 0.1, True, True)
        controller.frame = 100
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., -0.02, 0.])
        control.actuators.accel = request
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=22., aEgo=-0.05, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=119, odyssey_target_gear=7,
                             odyssey_user_brake=0., odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(100, 302):
          if frame == 300:
            control.actuators.accel = 0.15
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          if frame % 2:
            continue
          signals = parser.vl['ACC_CONTROL']
          self.assertEqual(signals['BRAKE_REQUEST'], frame < 300)
          self.assertAlmostEqual(signals['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
          if frame < 300:
            self.assertEqual(signals['GAS_COMMAND'], -30000)
          else:
            self.assertGreater(signals['GAS_COMMAND'], 0)

  def test_nonnegative_brake_hold_requires_qualified_coast(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    faults = ('invalid_can', 'missing_gear', 'stale_gear', 'future_gear', 'missing_brake', 'stale_brake',
              'future_brake', 'missing_gas', 'stale_gas', 'future_gas', 'pose', 'gas_pedal', 'brake_pedal',
              'inactive', 'stopping', 'low_speed', 'high_speed', 'no_history', 'unknown_gear', 'gear_change', 'forecast', 'no_brake')
    for fault in (None, *faults):
      with self.subTest(fault=fault):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.odyssey_brake_selected = fault != 'no_brake'
        controller.odyssey_pitch.x = -0.03
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.drag[7] = 0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.03)
        controller.odyssey_brake_response_error.x = .3
        control = structs.CarControl(enabled=True, longActive=fault != 'inactive', orientationNED=[0., -0.03, 0.])
        control.actuators.accel = .05
        control.actuators.longControlState = (structs.CarControl.Actuators.LongControlState.stopping if fault == 'stopping'
                                              else structs.CarControl.Actuators.LongControlState.pid)
        state = structs.CarState(vEgo=20., aEgo=0., canValid=fault != 'invalid_can',
                                 gasPressed=fault == 'gas_pedal', brakePressed=fault == 'brake_pedal')
        now = 1_000_000_000
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_engine_torque_ts_nanos=now - 5_000_000, odyssey_shift_activity=0,
                             odyssey_target_gear=7, odyssey_target_gear_ts_nanos=now - 5_000_000,
                             odyssey_user_brake=0., odyssey_computer_braking=True, odyssey_computer_braking_ts_nanos=now - 5_000_000)
        if fault == 'pose':
          control.orientationNED = []
        elif fault in ('low_speed', 'high_speed'):
          state.vEgo = 7.9 if fault == 'low_speed' else 35.1
        elif fault == 'no_history':
          controller.odyssey_coast_response.drag.clear()
        elif fault == 'unknown_gear':
          CS.odyssey_target_gear = 8
        elif fault == 'gear_change':
          CS.odyssey_target_gear = 8
          controller.odyssey_coast_response.drag[8] = controller.odyssey_coast_response.drag[7]
        elif fault == 'forecast':
          controller.odyssey_coast_response.drag[7] -= .3
        elif fault and fault.split('_')[0] in ('missing', 'stale', 'future'):
          age, signal = fault.split('_')
          name = {'gear': 'odyssey_target_gear_ts_nanos', 'brake': 'odyssey_computer_braking_ts_nanos',
                  'gas': 'odyssey_engine_torque_ts_nanos'}[signal]
          setattr(CS, name, {'missing': 0, 'stale': now - 60_000_000, 'future': now + 1}[age])
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], fault in (None, 'gear_change'))
        if fault == 'gear_change':
          self.assertLessEqual(controller.odyssey_brake_response_error.x, .05)
        elif fault not in (None, 'forecast'):
          self.assertEqual(controller.odyssey_brake_response_error.x, 0.)

  def test_brake_hold_ignores_a_transient_acceleration_shortfall(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for eco_mode in (False, True):
      with self.subTest(eco_mode=eco_mode):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_pitch.x = -0.02
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 50
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.drag[7] = 0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.02)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., -0.02, 0.])
        control.actuators.accel = .15
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=20., aEgo=.15, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=119, odyssey_target_gear=7,
                             odyssey_user_brake=.015625, odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        # A brief wheel-derived acceleration dip must not interrupt a followed, positive brake hold.
        for aego in [.15] * 100 + [-.15] * 2 + [.15] * 100:
          state.aEgo = aego
          now = 1_000_000_000 + controller.frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 1)
          self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
          self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], .15, delta=.01)

  def test_brake_hold_yields_to_positive_acceleration_shortfall(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for eco_mode, blocked in product((False, True),
                                    (None, 'braking', 'pressure', 'missing_pressure', 'gas', 'settling',
                                     'missing_start', 'response', 'zero', 'negative')):
      with self.subTest(eco_mode=eco_mode, blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_pitch.x = -0.02
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.drag[7] = 0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.02)
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = None if blocked == 'missing_start' else 52 if blocked == 'settling' else 50
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., -0.02, 0.])
        control.actuators.accel = {'zero': 0., 'negative': -0.1}.get(blocked, .15)
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=20., aEgo=.1 if blocked == 'response' else control.actuators.accel - .3,
                                 canValid=True)
        now = 1_000_000_000
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=1. if blocked == 'gas' else 0., odyssey_shift_activity=119, odyssey_target_gear=7,
                             odyssey_user_brake={'pressure': .03125, 'missing_pressure': math.nan}.get(blocked, .015625),
                             odyssey_computer_braking=blocked == 'braking', odyssey_engine_torque_ts_nanos=now - 5_000_000,
                             odyssey_target_gear_ts_nanos=now - 5_000_000, odyssey_computer_braking_ts_nanos=now - 5_000_000)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        # Sustained shortfall still releases; a single sample does not establish the response.
        for _ in range(1 if blocked in ('zero', 'negative') else 100):
          if blocked == 'settling':
            controller.odyssey_brake_start_frame = controller.frame - 48
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          messages = [msg for msg in sends if msg[0] == 0x1DF]
          parser.update([(now, messages)])
          signals = parser.vl['ACC_CONTROL']
          if not signals['BRAKE_REQUEST']:
            break
          now += 10_000_000
        released = blocked in (None, 'braking', 'pressure')
        self.assertEqual(signals['BRAKE_REQUEST'], not released)
        self.assertEqual(signals['GAS_COMMAND'] == -30000, blocked is not None)
        self.assertAlmostEqual(signals['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
        self.assertAlmostEqual(output.accel, control.actuators.accel, delta=.01)
        self.assertAlmostEqual(controller.odyssey_coast_response.drag[7],
                               0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.02))
        if released:
          checked_messages = list(messages)
          if blocked is None:
            self.assertIsNone(controller.odyssey_brake_start_frame)
          else:
            self.assertEqual(controller.odyssey_brake_start_frame, 50)
            # Releasing the command must still wait for received brake feedback before gas.
            CS.odyssey_computer_braking = False
            CS.odyssey_user_brake = 0.
            for _ in range(2):
              now += 10_000_000
              CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
              _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
              messages = [msg for msg in sends if msg[0] == 0x1DF]
              checked_messages.extend(messages)
              parser.update([(now, messages)])
            self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
            self.assertGreater(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
          safety = libsafety_py.libsafety
          config = CP.safetyConfigs[-1]
          self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
          safety.init_tests()
          safety.set_controls_allowed(True)
          for addr, dat, bus in checked_messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))

  def test_first_coast_estimate_does_not_import_code_in_the_control_loop(self):
    result = subprocess.run([sys.executable, "-c", """
import sys
from opendbc.car.honda.carcontroller import OdysseyCoastResponse

response = OdysseyCoastResponse()
for frame in range(0, 58, 2):
  response.update(frame, 7, 0.0, -0.2, True, True)
assert len(response.warmup[7]) == 4
loaded = set(sys.modules)
assert response.update(58, 7, 0.0, -0.2, True, True) is None
assert not set(sys.modules) - loaded, sorted(set(sys.modules) - loaded)
assert response.update(60, 7, 0.0, -0.2, True, True) == -0.2
"""], capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_passive_coast_forecast_tracks_changed_response_without_learning_an_outlier(self):
    response = OdysseyCoastResponse()
    for frame in range(0, 100, 2):
      response.update(frame, 7, 0.0, -0.15, True, True)
    self.assertAlmostEqual(response.update(100, 7, 0.0, 0.30, True, True), -0.15)
    for frame in range(102, 140, 2):
      response.update(frame, 7, 0.0, 0.30, True, True)
    self.assertAlmostEqual(response.update(140, 7, 0.0, 3.0, True, True), 0.30)
    self.assertAlmostEqual(response.update(142, 7, 0.0, 0.30, True, True), 0.30)
    for frame in range(144, 210, 2):
      self.assertAlmostEqual(response.update(frame, 7, 0.0, -3.0, False, True), 0.30)

  def test_incomplete_coast_samples_cannot_bridge_interrupted_observations(self):
    for interruption in ('controlled', 'gear', 'invalid'):
      with self.subTest(interruption=interruption):
        response = OdysseyCoastResponse()
        for frame in range(0, 56, 2):
          self.assertIsNone(response.update(frame, 7, 0., -.4, True, True))
        self.assertEqual(len(response.warmup[7]), 3)
        response.update(56, 8 if interruption == 'gear' else 7, 0., -.4,
                        interruption != 'controlled', interruption != 'invalid')
        for frame in range(3600, 3660, 2):
          self.assertIsNone(response.update(frame, 7, 0., 3. if frame == 3658 else .1, True, True))
        self.assertAlmostEqual(response.update(3660, 7, 0., .1, True, True), .1)
        self.assertAlmostEqual(response.update(3662, 7, 0., 3., False, True), .1)

  def test_new_coast_uses_fresh_observations_while_retaining_its_prior_forecast(self):
    response = OdysseyCoastResponse()
    for frame in range(0, 100, 2):
      response.update(frame, 7, 0., -.4, True, True)
    self.assertAlmostEqual(response.update(100, 7, 0., 3., False, True), -.4)
    for frame in range(200, 292, 2):
      self.assertEqual(response.update(frame, 7, 0., .1, True, True), -.4)
    self.assertAlmostEqual(response.update(292, 7, 0., .1, True, True), .1)

  def test_downhill_gas_release_requires_response_and_preserves_brake_authority(self):
    response = OdysseyGasCoastRelease()
    for frame in range(0, 20, 2):
      self.assertFalse(response.update(frame * 10_000_000, 0.10, 0.30, 25.0, -0.03, 9, 0.09, True, True))
    for frame in range(20, 40, 2):
      self.assertFalse(response.update(frame * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, None, True, True))
    self.assertTrue(response.update(40 * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, 0.09, True, True))
    self.assertEqual(odyssey_command_domains(-0.15, 25.0, previous_gas=True, release_gas=True), (False, False))
    self.assertEqual(odyssey_command_domains(-0.40, 25.0, previous_gas=True, release_gas=True), (False, True))
    self.assertEqual(odyssey_command_domains(-0.15, 4.0, previous_gas=True, release_gas=True), (False, True))
    self.assertTrue(response.update(42 * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, 0.09, False, True))
    self.assertFalse(response.update(44 * 10_000_000, -0.15, -0.11, 25.0, -0.03, 9, 0.09, False, True))
    self.assertFalse(response.update(46 * 10_000_000, 0.10, 0.30, 25.0, -0.03, 9, 0.09, True, True))
    self.assertFalse(response.update(48 * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, 0.09, True, False))
    self.assertFalse(response.requests)

  def test_downhill_coast_requires_learned_response_or_prior_gas(self):
    response = OdysseyGasCoastRelease()
    for frame in range(0, 20, 2):
      response.update(frame * 10_000_000, 0.10, 0.30, 25.0, -0.03, 9, 0.09, True, True)
    self.assertFalse(response.update(20 * 10_000_000, -0.15, 0.30, 25.0, 0.03, 9, 0.09, True, True))
    self.assertFalse(response.update(22 * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, None, False, True))
    self.assertTrue(response.update(23 * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, 0.09, False, True))
    self.assertFalse(response.update(24 * 10_000_000, -0.15, 0.30, 25.0, -0.03, 9, 0.09, True, False))

  def test_downhill_coast_follows_positive_demand_and_measured_response(self):
    for request, aego, passive in ((0.35, 0.60, 0.30), (0.05, 0.06, 0.30), (0.05, 0.60, 0.0)):
      response = OdysseyGasCoastRelease()
      self.assertTrue(response.update(0, 0.05, 0.60, 20.0, -0.04, 7, 0.30, True, True))
      self.assertEqual(odyssey_command_domains(0.05, 20.0, previous_gas=True, release_gas=True), (False, False))
      self.assertFalse(response.update(2 * 10_000_000, request, aego, 20.0, -0.04, 7, passive, False, True))
      self.assertEqual(odyssey_command_domains(request, 20.0, release_gas=response.active), (True, False))
    for passive in (None, float("inf"), float("-inf"), float("nan")):
      response = OdysseyGasCoastRelease()
      self.assertFalse(response.update(0, 0.05, 0.60, 20.0, -0.04, 7, passive, True, True, True))

  def test_learned_coast_does_not_pre_activate_gas_below_passive_acceleration(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    CI = CarInterface(CP.copy(), _sp(CP))
    CI.update([])
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    for frame in range(0, 102, 2):
      controller.odyssey_coast_response.update(frame, 7, 0.0, -0.1, True, True)
    controller.frame = 102
    controller.odyssey_pitch.x = -0.04
    CI.CS.out = structs.CarState(vEgo=20.0, vEgoRaw=20.0, aEgo=0.30, canValid=True)
    CI.CS.odyssey_target_gear = 7
    CI.CS.odyssey_car_gas = 0.0
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0.0, -0.04, 0.0],
                                 actuators=structs.CarControl.Actuators(accel=0.05, longControlState=structs.CarControl.Actuators.LongControlState.pid))
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    seen = 0
    for frame in range(6):
      now = 1_000_000_000 + frame * 10_000_000
      CI.CS.odyssey_engine_torque_ts_nanos = now - 5_000_000
      CI.CS.odyssey_target_gear_ts_nanos = now - 5_000_000
      CI.CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CI.CS, now)
      if any(addr == 0x1DF for addr, _, _ in sends):
        parser.update([(now, sends)])
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 1)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], 0.05)
        seen += 1
    self.assertEqual(seen, 3)
    for frame in range(4):
      now = 1_100_000_000 + frame * 10_000_000
      CI.CS.out.canValid = frame >= 2
      CI.CS.odyssey_engine_torque_ts_nanos = now - 5_000_000
      CI.CS.odyssey_target_gear_ts_nanos = now - (60_000_000 if frame >= 2 else 5_000_000)
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CI.CS, now)
      if any(addr == 0x1DF for addr, _, _ in sends):
        parser.update([(now, sends)])
        self.assertGreater(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], 0.05)

  def test_downhill_gas_floor_releases_without_coast_history_only_on_falling_request(self):
    def run(gas_floor, pitch=-0.04, aego=0.4, request=-0.12):
      response = OdysseyGasCoastRelease()
      for frame in range(0, 22, 2):
        response.update(frame * 10_000_000, 0.02, aego, 21.0, pitch, 7, None, True, True, gas_floor)
      return response.update(22 * 10_000_000, request, aego, 21.0, pitch, 7, None, True, True, gas_floor)

    self.assertTrue(run(True))
    self.assertFalse(run(False))
    self.assertFalse(run(True, pitch=0.04))
    self.assertFalse(run(True, aego=-0.1))
    self.assertFalse(run(True, request=0.02))

  def test_gas_coast_fallback_does_not_overstate_the_request_trend(self):
    for slope in (-0.19, -0.21):
      response = OdysseyGasCoastRelease()
      for frame in range(0, 42, 2):
        request = -0.05 + slope * frame * 0.01
        released = response.update(frame * 10_000_000, request, 0.4, 21.0, -0.04, 7, None, True, True, True)
        self.assertEqual(released, slope < -0.20 and frame >= 20)

    response = OdysseyGasCoastRelease()
    for frame in range(0, 22, 2):
      self.assertFalse(response.update(frame * 10_000_000, -0.05, 0.4, 21.0, -0.04, 7, None, True, True, True))
    self.assertFalse(response.update(100 * 10_000_000, -0.10, 0.4, 21.0, -0.04, 7, None, True, True, True))
    self.assertTrue(response.update(102 * 10_000_000, -0.29, 0.4, 21.0, -0.04, 7, None, True, True, True))

  def test_exhausted_gas_command_releases_to_coast_without_requesting_brake(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    control = structs.CarControl()
    control.enabled = True
    control.orientationNED = [0.0, -0.04, 0.0]
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo = 21.0
    state.aEgo = 0.5
    state.canValid = True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

    for frame in range(260):
      control.longActive = frame >= 100
      control.actuators.accel = 0.1 if frame < 220 else max(-0.14, 0.1 - (frame - 220) * 0.008)
      CS.odyssey_engine_torque_ts_nanos = frame * 10_000_000
      CS.odyssey_target_gear_ts_nanos = frame * 10_000_000
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, frame * 10_000_000)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if messages:
        parser.update([(frame * 10_000_000, messages)])
      if frame == 220:
        self.assertGreaterEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], 0)
      if frame == 240:
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)

    self.assertFalse(controller.odyssey_coast_response.drag)
    self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
    self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
    self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], -0.14, delta=0.01)

  def test_sustained_zero_gas_excess_releases_before_learning_coast_in_both_modes(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    for econ, mild_request in product((False, True), (-.05, 0., .05)):
      with self.subTest(econ=econ, request=mild_request):
        CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
        controller = CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, _sp(CP))
        state = structs.CarState(vEgo=20., aEgo=.4, canValid=True)
        CS = CarState(CP, _sp(CP))
        CS.update(CS.get_can_parsers(CP, _sp(CP)))
        CS.out = state
        CS.odyssey_car_gas = CS.odyssey_user_brake = 0.
        CS.odyssey_computer_braking = False
        CS.odyssey_target_gear = 7
        CS.odyssey_shift_activity = 119
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        parser = CANParser(DBC[CP.carFingerprint][Bus.pt], [('ACC_CONTROL', 0)], 1)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        first_release = None
        for frame in range(384):
          request = .05 if frame < 300 else mild_request if frame < 370 else .6 if frame < 380 else -.4
          control.actuators.accel = request
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          previous_wire_gas = parser.vl['ACC_CONTROL']['GAS_COMMAND']
          history = len(controller.odyssey_gas_response.requests)
          filtered_error = controller.odyssey_gas_response.error.x
          prior_coast = controller.odyssey_coast_response.drag.get(7)
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          if frame % 2:
            continue
          packets = [msg for msg in sends if msg[0] == 0x1DF]
          for addr, dat, bus in packets:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))
          parser.update([(now, packets)])
          signals = parser.vl['ACC_CONTROL']
          self.assertAlmostEqual(signals['ACCEL_COMMAND'], request, delta=.005)
          self.assertAlmostEqual(output.accel, request, delta=.005)
          if signals['GAS_COMMAND'] == -30000 and first_release is None:
            first_release = frame
            self.assertEqual(previous_wire_gas, 0)
            self.assertGreater(history, ODYSSEY_RESPONSE_DELAY_FRAMES)
            self.assertLess(filtered_error, -.2)
          if first_release is None or frame < first_release + 2 * ODYSSEY_RESPONSE_DELAY_FRAMES:
            self.assertFalse(controller.odyssey_coast_response.drag)
          if frame == 298:
            self.assertIsNotNone(first_release)
            self.assertTrue(controller.odyssey_gas_coast_release.exhausted)
            self.assertEqual((signals['GAS_COMMAND'], signals['BRAKE_REQUEST']), (-30000, 0))
          if frame < 370 and prior_coast is None:
            self.assertEqual(signals['BRAKE_REQUEST'], 0)
          if frame == 368 or frame >= 380:
            self.assertEqual((signals['GAS_COMMAND'], signals['BRAKE_REQUEST']), (-30000, 1))
          elif 370 <= frame < 380:
            self.assertGreater(signals['GAS_COMMAND'], 0)
            self.assertEqual(signals['BRAKE_REQUEST'], 0)

  def test_zero_gas_release_rejects_unreliable_or_unsettled_response(self):
    faults = (None, 'can', 'gas_age', 'gear_age', 'brake_age', 'received_gas', 'received_brake', 'pressure',
              'invalid_pressure', 'gas_pedal', 'brake_pedal', 'pose', 'inactive', 'stopping', 'low_speed', 'high_speed',
              'gear', 'activity', 'short_history', 'small_error', 'latest_response', 'issued_gas', 'bridge', 'new_domain')
    encoded_commands = (-60., -1., math.nextafter(-.5, -math.inf), -.5, math.nextafter(-.5, math.inf), 0., .25, .49,
                        math.nextafter(math.nextafter(.5, -math.inf), -math.inf), math.nextafter(.5, -math.inf),
                        .5, math.nextafter(.5, math.inf), 1.)
    for request, (fault, issued) in product((-.05, 0., .05),
                                           [(fault, None) for fault in faults] + [('encoding', gas) for gas in encoded_commands]):
      with self.subTest(request=request, fault=fault, issued=issued):
        CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
        controller = CarController({Bus.pt: DBC[CP.carFingerprint][Bus.pt]}, CP, _sp(CP))
        CS = CarState(CP, _sp(CP))
        CS.update(CS.get_can_parsers(CP, _sp(CP)))
        CS.out = structs.CarState(vEgo=20., aEgo=.4, canValid=True)
        CS.odyssey_car_gas, CS.odyssey_user_brake = 1., 0.
        CS.odyssey_computer_braking = False
        CS.odyssey_target_gear = 7
        CS.odyssey_shift_activity = 119
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = .05
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        parser = CANParser(DBC[CP.carFingerprint][Bus.pt], [('ACC_CONTROL', 0)], 1)
        expected_release = fault is None
        for frame in range(301):
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          if frame == 300:
            CS.odyssey_car_gas = 0.
            control.actuators.accel = request
            if issued is not None:
              controller.gas = issued
            packet = CANPacker(DBC[CP.carFingerprint][Bus.pt]).make_can_msg('ACC_CONTROL', 1, {'GAS_COMMAND': controller.gas})
            parser.update([(now - 20_000_000, [packet])])
            wire_gas = parser.vl['ACC_CONTROL']['GAS_COMMAND']
            if issued is None:
              self.assertEqual(wire_gas, 0.)
            else:
              expected_release = wire_gas == 0.
            if fault == 'can':
              CS.out.canValid = False
            elif fault in ('gas_age', 'gear_age', 'brake_age'):
              setattr(CS, {'gas_age': 'odyssey_engine_torque_ts_nanos', 'gear_age': 'odyssey_target_gear_ts_nanos',
                           'brake_age': 'odyssey_computer_braking_ts_nanos'}[fault], now - 60_000_000)
            elif fault == 'received_gas':
              CS.odyssey_car_gas = 1.
            elif fault == 'received_brake':
              CS.odyssey_computer_braking = True
            elif fault in ('pressure', 'invalid_pressure'):
              CS.odyssey_user_brake = .03125 if fault == 'pressure' else math.nan
            elif fault in ('gas_pedal', 'brake_pedal'):
              setattr(CS.out, 'gasPressed' if fault == 'gas_pedal' else 'brakePressed', True)
            elif fault == 'pose':
              control.orientationNED = []
            elif fault == 'inactive':
              control.longActive = False
            elif fault == 'stopping':
              control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
            elif fault in ('low_speed', 'high_speed'):
              CS.out.vEgo = 7.99 if fault == 'low_speed' else 35.01
            elif fault == 'gear':
              CS.odyssey_target_gear = 8
            elif fault == 'activity':
              CS.odyssey_shift_activity = 126
            elif fault == 'short_history':
              controller.odyssey_gas_response.clear_observer()
            elif fault == 'small_error':
              controller.odyssey_gas_response.error.x = -.19
            elif fault == 'latest_response':
              CS.out.aEgo = -.1
            elif fault in ('issued_gas', 'bridge'):
              controller.gas = 1. if fault == 'issued_gas' else -60.
            elif fault == 'new_domain':
              controller.odyssey_gas_selected = False
          controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        self.assertEqual(controller.odyssey_gas_coast_release.active, expected_release)
        self.assertEqual(controller.odyssey_gas_coast_release.exhausted, expected_release)

  def test_exhausted_coast_yields_to_response_request_or_invalid_state(self):
    for request, forecast_offset in product((-.1, 0., .1), (None, -.3, .02, .5)):
      forecast = None if forecast_offset is None else request + forecast_offset
      with self.subTest(request=request, forecast=forecast):
        release = OdysseyGasCoastRelease()
        self.assertTrue(release.update(0, request, request + .3, 20., 0., 7, forecast, True, True, True, True, False))
        for frame in range(1, 30):
          excess = .3 if frame < 10 else .19 if frame < 20 else .06
          self.assertTrue(release.update(frame * 20_000_000, request, request + excess, 20., 0., 7, forecast, False, True,
                                         downhill=False))
        now = 600_000_000
        for next_request, achieved, speed, gear, eligible in (
          (request, request + .04, 20., 7, True),
          (-.4, request + .3, 20., 7, True),
          (request, request + .3, 7.99, 7, True),
          (request, request + .3, 35.01, 7, True),
          (request, request + .3, 20., 8, True),
          (request, request + .3, 20., 7, False),
        ):
          self.assertTrue(release.update(now, request, request + .3, 20., 0., 7, forecast, True, True, True, True, False))
          self.assertFalse(release.update(now + 20_000_000, next_request, achieved, speed, 0., gear, forecast, False, eligible,
                                          downhill=False))
          now += 40_000_000

  def test_downhill_coast_waits_for_the_issued_zero_gas_command(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request, achieved, pitch, next_request, expected in (
      (.05, .5, -.03, -.01, (-30000, -30000)),
      (.5, .9, -.03, -.01, (0, -30000)),
      (-.001, .5, -.0256, -.048, (-60, -60)),
    ):
      with self.subTest(request=request):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.odyssey_pitch.x = pitch
        state = structs.CarState(vEgo=21., aEgo=achieved, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_target_gear=7, odyssey_shift_activity=119,
                             odyssey_user_brake=0., odyssey_computer_braking=False)
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., pitch, 0.])
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        observed = []
        for frame in range(154):
          control.actuators.accel = request if frame < 150 else next_request
          # Isolate the issued-command edge from earlier idle-gas exhaustion.
          CS.odyssey_car_gas = float(frame < 150)
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, sends)])
          if frame >= 150 and frame % 2 == 0:
            signals = parser.vl['ACC_CONTROL']
            observed.append(signals['GAS_COMMAND'])
            self.assertEqual(signals['BRAKE_REQUEST'], 0)
            self.assertAlmostEqual(signals['ACCEL_COMMAND'], next_request, delta=.005)
        self.assertFalse(controller.odyssey_coast_response.drag)
        self.assertEqual(tuple(observed), expected)

  def test_manual_coast_estimate_is_ready_at_longitudinal_engagement(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for blocked in (None, 'gas_pedal', 'brake_pedal', 'received_gas', 'computer_brake', 'pressure', 'missing_pressure',
                    'stale_gas', 'stale_gear', 'stale_brake', 'invalid_can', 'invalid_pose', 'invalid_gear',
                    'low_speed', 'high_speed', 'unsettled', 'active_non_pid'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        control = structs.CarControl(enabled=True, longActive=blocked == 'active_non_pid', orientationNED=[0., -.02, 0.])
        state = structs.CarState(vEgo={'low_speed': 7.9, 'high_speed': 35.1}.get(blocked, 20.),
                                 aEgo=.3, canValid=blocked != 'invalid_can',
                                 gasPressed=blocked == 'gas_pedal', brakePressed=blocked == 'brake_pedal')
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=1. if blocked == 'received_gas' else 0., odyssey_shift_activity=119,
                             odyssey_target_gear=0 if blocked == 'invalid_gear' else 7,
                             odyssey_user_brake={'pressure': .03125, 'missing_pressure': math.nan}.get(blocked, .015625),
                             odyssey_computer_braking=blocked == 'computer_brake')
        if blocked == 'invalid_pose':
          control.orientationNED = []
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)

        def step(stale=None, controller=controller, control=control, CS=CS, parser=parser):
          now = 1_000_000_000 + controller.frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = now - (61_000_000 if stale == 'stale_gas' else 5_000_000)
          CS.odyssey_target_gear_ts_nanos = now - (61_000_000 if stale == 'stale_gear' else 5_000_000)
          CS.odyssey_computer_braking_ts_nanos = now - (61_000_000 if stale == 'stale_brake' else 5_000_000)
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          messages = [msg for msg in sends if msg[0] == 0x1DF]
          parser.update([(now, messages)])
          return output, messages

        for _ in range(20 if blocked == 'unsettled' else 200):
          _, messages = step(blocked)
          if messages:
            signals = parser.vl['ACC_CONTROL']
            self.assertEqual(signals['BRAKE_REQUEST'], 0)
            self.assertEqual(signals['GAS_COMMAND'], -30000)
            self.assertEqual(signals['ACCEL_COMMAND'], 0)
        self.assertEqual(bool(controller.odyssey_coast_response.drag), blocked is None)
        control.longActive = True
        control.orientationNED = [0., -.02, 0.]
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        control.actuators.accel = -.1
        CS.out = structs.CarState(vEgo=20., aEgo=.3, canValid=True)
        CS.odyssey_car_gas = 0.
        CS.odyssey_target_gear = 7
        CS.odyssey_user_brake = .015625
        CS.odyssey_computer_braking = False
        output, messages = step()
        self.assertEqual(len(messages), 1)
        signals = parser.vl['ACC_CONTROL']
        self.assertEqual(signals['BRAKE_REQUEST'], blocked is None)
        self.assertAlmostEqual(signals['ACCEL_COMMAND'], -.1, delta=.01)
        self.assertAlmostEqual(output.accel, -.1, delta=.01)
        if blocked is None:
          self.assertEqual(signals['GAS_COMMAND'], -30000)
          safety = libsafety_py.libsafety
          config = CP.safetyConfigs[-1]
          self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
          safety.init_tests()
          safety.set_controls_allowed(True)
          for addr, dat, bus in messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))

  def test_coast_estimate_does_not_learn_changing_grade_as_drag(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    control = structs.CarControl(orientationNED=[0., 0., 0.])
    state = structs.CarState(vEgo=20., canValid=True)
    CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0., odyssey_shift_activity=119, odyssey_target_gear=7,
                         odyssey_user_brake=0., odyssey_computer_braking=False)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    drag = -.18
    for frame in range(800):
      # Passive acceleration changes with gravity while drivetrain drag stays fixed.
      pitch = 0. if frame < 200 else -.00025 * (frame - 200) if frame < 400 else -.05 + .00025 * (frame - 400)
      control.orientationNED = [0., pitch, 0.]
      state.aEgo = -ACCELERATION_DUE_TO_GRAVITY * math.sin(pitch) + drag
      state.vEgo += state.aEgo * DT_CTRL
      now = 1_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      messages = [msg for msg in sends if msg[0] == 0x1DF]
      if messages:
        parser.update([(now, messages)])
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 0)
        self.assertEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], 0.)
      if frame >= 200:
        self.assertAlmostEqual(controller.odyssey_coast_response.drag[7], drag)

  def test_settled_coast_overdeceleration_enters_mapped_gas_and_retains_brake_authority(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    safety = libsafety_py.libsafety
    config = CP.safetyConfigs[-1]
    self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
    safety.init_tests()
    safety.set_controls_allowed(True)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, 0.03, 0.0]
    control.actuators.accel = -0.18
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.aEgo = 21.0, -0.45
    state.canValid = True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)

    def step():
      frame = controller.frame
      now = 1_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = now
      CS.odyssey_target_gear_ts_nanos = now
      CS.odyssey_computer_braking_ts_nanos = now
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if messages:
        for addr, dat, src in messages:
          self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, src, dat)))
        parser.update([(now, messages)])
      return parser.vl['ACC_CONTROL']

    for _ in range(60):
      signals = step()
      self.assertEqual(signals['GAS_COMMAND'], -30000)
    for frame in range(60, 130):
      signals = step()
      if frame >= 60:
        self.assertGreater(signals['GAS_COMMAND'], 0)
        self.assertEqual(signals['BRAKE_REQUEST'], 0)
        self.assertAlmostEqual(signals['ACCEL_COMMAND'], -0.18, delta=0.01)
        self.assertFalse(controller.odyssey_gas_bridge_active)
    control.actuators.accel = -0.6
    signals = step()
    self.assertEqual(signals['GAS_COMMAND'], -30000)
    self.assertEqual(signals['BRAKE_REQUEST'], 1)
    control.actuators.accel = 0.1
    step()
    signals = step()
    self.assertGreater(signals['GAS_COMMAND'], 0)
    self.assertEqual(signals['BRAKE_REQUEST'], 0)

  def test_received_brake_signal_preserves_idle_quantization_and_timestamp(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    CS = CarState(CP, _sp(CP))
    self.assertTrue(math.isnan(CS.odyssey_user_brake))
    packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])
    parsers = CS.get_can_parsers(CP, _sp(CP))
    CS.update(parsers)
    for i, value in enumerate((.125, .015625, 0., -.015625)):
      now = 1_000_000_000 + i * 20_000_000
      parsers[Bus.pt].update([(now, [packer.make_can_msg('VSA_STATUS', 1,
                                                       {'USER_BRAKE': value, 'COMPUTER_BRAKING': i == 0, 'COUNTER': i})])])
      CS.update(parsers)
      self.assertEqual(CS.odyssey_user_brake, value)
      self.assertEqual(CS.odyssey_computer_braking, i == 0)
      self.assertEqual(CS.odyssey_computer_braking_ts_nanos, now)

  def test_held_brake_overdeceleration_recovers_gas_without_coast_history(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for blocked in (None, 'braking', 'gas', 'intermittent_braking', 'pending', 'stale_brake', 'stale_gear',
                    'gas_pedal', 'brake_pedal', 'invalid_CAN', 'pose', 'pressure', 'missing_pressure'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.odyssey_pitch.x = 0.04
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[] if blocked == 'pose' else [0., .04, 0.])
        control.actuators.accel = -0.4
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=21., aEgo=-0.8, canValid=blocked != 'invalid_CAN',
                                 gasPressed=blocked == 'gas_pedal', brakePressed=blocked == 'brake_pedal')
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=1. if blocked == 'gas' else 0., odyssey_target_gear=7, odyssey_shift_activity=119,
                             odyssey_user_brake={'pressure': .0625, 'missing_pressure': math.nan}.get(blocked, .015625))
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(220):
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = now - 5_000_000
          CS.odyssey_target_gear_ts_nanos = now - (80_000_000 if blocked == 'stale_gear' else 5_000_000)
          CS.odyssey_computer_braking_ts_nanos = now - (80_000_000 if blocked == 'stale_brake' else 5_000_000)
          CS.odyssey_computer_braking = blocked == 'braking' or (blocked == 'intermittent_braking' and frame % 40 == 0)
          if blocked == 'pending':
            controller.odyssey_brake_start_frame = frame
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          signals = parser.vl['ACC_CONTROL']
          if blocked is not None or frame < 100:
            self.assertEqual(signals['GAS_COMMAND'], -30000)
            self.assertEqual(signals['BRAKE_REQUEST'], 1)
            self.assertFalse(controller.odyssey_coast_response.drag)
          elif frame >= 150:
            self.assertGreaterEqual(signals['GAS_COMMAND'], 0)
            self.assertEqual(signals['BRAKE_REQUEST'], 0)
            self.assertAlmostEqual(signals['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
            self.assertFalse(controller.odyssey_gas_bridge_active)
            if frame >= 180:
              self.assertGreater(signals['GAS_COMMAND'], 0)

  def test_brake_release_does_not_reselect_brakes_while_waiting_for_idle_feedback(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for tail_frames, change, early_release in ((20, None, False), (120, None, False), (20, 'gear', False),
                                                (20, 'stronger', False), (20, 'inactive', False),
                                                (20, None, True), (20, 'gear', True)):
      with self.subTest(tail_frames=tail_frames, change=change, early_release=early_release):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        controller.odyssey_pitch.x = 0.04
        controller.odyssey_coast_response.gear = 7
        drag = -0.8 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.04)
        controller.odyssey_coast_response.drag[7] = drag
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0., 0.04, 0.]
        control.actuators.accel = -0.4
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=22., aEgo=-0.9, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_user_brake=0.125, odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(100, 101 + tail_frames):
          now = 2_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
          released = frame == 100 + tail_frames
          CS.odyssey_computer_braking = early_release and frame < 102
          if released:
            CS.odyssey_user_brake = 0.015625
            state.aEgo = -0.8
            if change == 'gear':
              CS.odyssey_target_gear = 8
              controller.odyssey_coast_response.drag[8] = drag
            elif change == 'stronger':
              control.actuators.accel = -0.95
            elif change == 'inactive':
              control.longActive = False
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          messages = [(addr, dat, bus) for addr, dat, bus in sends if addr == 0x1DF and bus == 1]
          if not messages:
            continue
          parser.update([(now, messages)])
          values = parser.vl['ACC_CONTROL']
          self.assertAlmostEqual(values['ACCEL_COMMAND'], control.actuators.accel if control.longActive else 0., delta=.01)
          self.assertEqual(controller.odyssey_coast_response.drag[7], drag)
          self.assertFalse(controller.odyssey_coast_response.warmup)
          if released and change is None:
            self.assertGreaterEqual(values['GAS_COMMAND'], 0)
            self.assertEqual(values['BRAKE_REQUEST'], 0)
            self.assertFalse(controller.odyssey_gas_bridge_active)
          else:
            self.assertEqual(values['GAS_COMMAND'], -30000)
            self.assertEqual(values['BRAKE_REQUEST'], released and change == 'stronger')
          if released and change == 'stronger':
            self.assertEqual(controller.odyssey_brake_start_frame, frame)
        if change == 'gear':
          # The learned new-gear coast can avoid braking, but cannot bypass gas-entry settling.
          state.aEgo = -0.1
          for _ in range(2):
            now += 10_000_000
            CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
            _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
            parser.update([(now, [(addr, dat, bus) for addr, dat, bus in sends if addr == 0x1DF and bus == 1])])
          self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
          self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 1)

  def test_coast_gas_entry_requires_settled_reliable_underresponse(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for blocked in ('settling', 'no_history', 'stale', 'pose', 'gas_pedal', 'brake_pedal', 'inactive',
                    'not_pid', 'low_speed', 'high_speed', 'gear', 'forecast', 'response',
                    'invalid_CAN', 'missing_gear', 'stale_gear', 'missing_VSA', 'stale_VSA', 'received_gas', 'received_brake'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_pitch.x = 0.03
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.coast_start = 0
        controller.odyssey_coast_response.drag[7] = -0.45 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.03)
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0.0, 0.03, 0.0]
        control.actuators.accel = -0.18
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState()
        state.vEgo, state.aEgo = 21.0, -0.45
        state.canValid = True
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0.0,
                             odyssey_engine_torque_ts_nanos=1_000_000_000, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=1_000_000_000,
                             odyssey_target_gear_ts_nanos=1_000_000_000)
        if blocked == 'settling':
          controller.odyssey_coast_response.coast_start = 98
        elif blocked == 'no_history':
          controller.odyssey_coast_response.drag.clear()
        elif blocked == 'stale':
          CS.odyssey_engine_torque_ts_nanos -= 80_000_000
        elif blocked == 'pose':
          control.orientationNED = []
        elif blocked == 'gas_pedal':
          state.gasPressed = True
        elif blocked == 'brake_pedal':
          state.brakePressed = True
        elif blocked == 'inactive':
          control.longActive = False
        elif blocked == 'not_pid':
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
        elif blocked == 'low_speed':
          state.vEgo = 4.0
        elif blocked == 'gear':
          CS.odyssey_target_gear = 8
          controller.odyssey_coast_response.drag[8] = controller.odyssey_coast_response.drag[7]
        elif blocked == 'forecast':
          controller.odyssey_coast_response.drag[7] = -0.3 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.03)
        elif blocked == 'response':
          state.aEgo = -0.1
        elif blocked == 'high_speed':
          state.vEgo = 40.0
        elif blocked == 'invalid_CAN':
          state.canValid = False
        elif blocked == 'missing_gear':
          CS.odyssey_target_gear_ts_nanos = 0
        elif blocked == 'stale_gear':
          CS.odyssey_target_gear_ts_nanos -= 80_000_000
        elif blocked == 'missing_VSA':
          CS.odyssey_computer_braking_ts_nanos = 0
        elif blocked == 'stale_VSA':
          CS.odyssey_computer_braking_ts_nanos -= 80_000_000
        elif blocked == 'received_gas':
          CS.odyssey_car_gas = 1.0
          controller.odyssey_brake_selected = True
        elif blocked == 'received_brake':
          CS.odyssey_computer_braking = True
          controller.odyssey_brake_selected = True
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, 1_000_000_000)
        parser.update([(1_000_000_000, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], blocked in ('low_speed', 'received_gas'))

  def test_brake_entry_ignores_transient_passive_response_before_settling(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    controller.odyssey_pitch.x = 0.04
    controller.odyssey_coast_response.gear = 7
    controller.odyssey_coast_response.drag[7] = -0.8 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.04)
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, 0.04, 0.0]
    control.actuators.accel = -0.4
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.canValid = 22.0, True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0.0,
                         odyssey_shift_activity=0, odyssey_target_gear=7, odyssey_user_brake=0., odyssey_computer_braking=False)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    for frame in range(2 * ODYSSEY_RESPONSE_DELAY_FRAMES + 1):
      now = 2_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
      state.aEgo = -0.8 if 2 <= frame <= 6 or frame == 2 * ODYSSEY_RESPONSE_DELAY_FRAMES else -0.1
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if not messages:
        continue
      parser.update([(now, messages)])
      values = parser.vl['ACC_CONTROL']
      if frame < 2 * ODYSSEY_RESPONSE_DELAY_FRAMES:
        self.assertEqual(values['GAS_COMMAND'], -30000)
        self.assertEqual(values['BRAKE_REQUEST'], 1)
      else:
        self.assertGreaterEqual(values['GAS_COMMAND'], 0)
        self.assertEqual(values['BRAKE_REQUEST'], 0)

    for frame in range(51, 57):
      now = 2_000_000_000 + frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
      state.aEgo = -0.8 if frame >= 54 else -0.1
      control.actuators.accel = 0.1 if frame in (51, 56) else -0.4
      _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
      if not messages:
        continue
      parser.update([(now, messages)])
      values = parser.vl['ACC_CONTROL']
      # Settled gas authority survives a brief response transient.
      self.assertEqual(values['BRAKE_REQUEST'], 0)
      self.assertGreaterEqual(values['GAS_COMMAND'], 0)

  def test_passive_gas_floor_feedback_delay_continuity_and_brake_fallback(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for change in ('none', 'at_target', 'approaching_passive', 'transient', 'excess', 'stronger'):
      with self.subTest(change=change):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_pitch.x = 0.04
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.drag[7] = -0.8 + ACCELERATION_DUE_TO_GRAVITY * math.sin(0.04)
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0.0, 0.04, 0.0]
        control.actuators.accel = -0.4
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState()
        state.vEgo, state.aEgo, state.canValid = 22.0, -0.8, True
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0.0,
                             odyssey_shift_activity=0, odyssey_target_gear=7, odyssey_user_brake=0., odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        last_gas = 0
        first_positive = None
        first_excess_brake = None
        for frame in range(100, 222):
          now = 2_000_000_000 + (frame - 100) * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = now - 10_000_000
          CS.odyssey_target_gear_ts_nanos = now - 10_000_000
          CS.odyssey_computer_braking_ts_nanos = now - 10_000_000
          if frame >= 160:
            if change == 'at_target':
              state.aEgo = -0.4
            elif change == 'approaching_passive':
              control.actuators.accel = -0.7
              state.aEgo = -0.7
            elif change == 'excess':
              state.aEgo = 0.0
            elif change == 'transient':
              state.aEgo = 0.0 if frame < 172 else -0.8
            elif change == 'stronger':
              control.actuators.accel = -0.95
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          messages = [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1]
          if not messages:
            continue
          for addr, dat, src in messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, src, dat)))
          parser.update([(now, messages)])
          values = parser.vl['ACC_CONTROL']
          gas, brake = values['GAS_COMMAND'], values['BRAKE_REQUEST']
          self.assertFalse(gas > -30000 and brake)
          self.assertAlmostEqual(output.accel, values['ACCEL_COMMAND'], delta=0.01)
          if frame >= 160 and change == 'excess' and brake and first_excess_brake is None:
            first_excess_brake = frame
          if frame >= 160 and (change == 'stronger' or (change == 'excess' and first_excess_brake is not None)):
            self.assertEqual(gas, -30000)
            self.assertEqual(brake, 1)
          else:
            self.assertGreaterEqual(gas, 0)
            self.assertEqual(brake, 0)
            self.assertAlmostEqual(values['ACCEL_COMMAND'], control.actuators.accel, delta=0.01)
            self.assertFalse(controller.odyssey_gas_bridge_active)
            self.assertLessEqual(abs(gas - last_gas), ODYSSEY_RESPONSE_SLEW_COUNTS)
            self.assertLessEqual(gas, controller.params.BOSCH_GAS_LOOKUP_V[-1])
            self.assertLessEqual(controller.odyssey_gas_response.integral, ODYSSEY_RESPONSE_MAX_COUNTS)
            if frame < 152:
              self.assertEqual(gas, 0)
            if gas > 0 and first_positive is None:
              first_positive = frame
          last_gas = gas
        self.assertEqual(first_positive, 100 + 2 * (ODYSSEY_RESPONSE_DELAY_FRAMES + 1))
        if change == 'none':
          self.assertGreater(last_gas, 0)
        if change == 'excess':
          assert first_excess_brake is not None
          self.assertLessEqual(first_excess_brake, 160 + 2 * ODYSSEY_RESPONSE_DELAY_FRAMES + 2)

  def test_gas_continuity_does_not_cycle_on_transient_response(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request, blocked in product((-.26, -.30, -.31, -.5),
                                   (None, 'latest_error', 'activity', 'stronger', 'bridge', 'no_history', 'forecast', 'at_forecast', 'gear',
                                    'stale_engine', 'stale_gear', 'stale_VSA', 'invalid_CAN', 'pose', 'gas_pedal',
                                    'brake_pedal', 'inactive', 'not_pid', 'low_speed', 'high_speed', 'received_brake')):
      with self.subTest(request=request, blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_gas_selected = True
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.drag[7] = -0.55
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = -0.1
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=20., aEgo=-0.1, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_user_brake=0., odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)

        def step(frame, CS=CS, controller=controller, control=control, parser=parser, blocked=blocked):
          now = 2_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          if frame >= 160:
            if blocked == 'stale_engine':
              CS.odyssey_engine_torque_ts_nanos -= 60_000_000
            elif blocked == 'stale_gear':
              CS.odyssey_target_gear_ts_nanos -= 60_000_000
            elif blocked == 'stale_VSA':
              CS.odyssey_computer_braking_ts_nanos -= 60_000_000
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          messages = [(addr, dat, bus) for addr, dat, bus in sends if addr == 0x1DF and bus == 1]
          parser.update([(now, messages)])
          values = parser.vl['ACC_CONTROL']
          self.assertAlmostEqual(values['ACCEL_COMMAND'], control.actuators.accel if control.longActive else 0., delta=.01)
          self.assertAlmostEqual(output.accel, values['ACCEL_COMMAND'], delta=.01)
          self.assertFalse(values['GAS_COMMAND'] > -30000 and values['BRAKE_REQUEST'])
          return values

        for frame in range(100, 160):
          step(frame)
        self.assertEqual(len(controller.odyssey_gas_response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
        control.actuators.accel, state.aEgo = request, 0.3
        if blocked in ('latest_error', 'activity'):
          controller.odyssey_gas_response.error.x = -1.
          if blocked == 'latest_error':
            state.aEgo = control.actuators.accel
          else:
            CS.odyssey_shift_activity = 1
        elif blocked == 'stronger':
          control.actuators.accel = -0.65
        elif blocked == 'bridge':
          controller.odyssey_gas_bridge_active = True
        elif blocked == 'no_history':
          controller.odyssey_coast_response.drag.clear()
        elif blocked == 'forecast':
          controller.odyssey_coast_response.drag[7] = -0.1
        elif blocked == 'at_forecast':
          controller.odyssey_coast_response.drag[7] = control.actuators.accel
        elif blocked == 'gear':
          CS.odyssey_target_gear = 8
        elif blocked == 'invalid_CAN':
          state.canValid = False
        elif blocked == 'pose':
          control.orientationNED = []
        elif blocked == 'gas_pedal':
          state.gasPressed = True
        elif blocked == 'brake_pedal':
          state.brakePressed = True
        elif blocked == 'inactive':
          control.longActive = False
        elif blocked == 'not_pid':
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
        elif blocked == 'low_speed':
          state.vEgo = 7.9
        elif blocked == 'high_speed':
          state.vEgo = 35.1
        elif blocked == 'received_brake':
          CS.odyssey_computer_braking = True
        for frame in range(160, 201):
          if blocked in (None, 'latest_error', 'activity') and frame == 172:
            state.aEgo = -0.6
          if blocked in (None, 'latest_error', 'activity') and frame == 200:
            control.actuators.accel, state.aEgo = -0.65, 0.3
          values = step(frame)
          if frame % 2:
            continue
          if blocked in (None, 'latest_error', 'activity') and frame < 200:
            self.assertGreaterEqual(values['GAS_COMMAND'], 0)
            self.assertEqual(values['BRAKE_REQUEST'], 0)
            if blocked != 'activity':
              self.assertEqual(len(controller.odyssey_gas_response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
            if frame <= 170:
              self.assertEqual(values['GAS_COMMAND'], 0)
            if frame == 198 and blocked is None:
              self.assertGreater(values['GAS_COMMAND'], 0)
          else:
            self.assertEqual(values['GAS_COMMAND'], -30000)
            self.assertEqual(values['BRAKE_REQUEST'], control.longActive and control.actuators.accel < ODYSSEY_ROAD_BRAKE_ENTRY)

  def test_mild_request_does_not_drop_needed_gas(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for change, eco_mode in product((None, 'at_target', 'more_decel', 'excess', 'stronger', 'gear', 'activity', 'stale_engine', 'stale_gear',
                                    'stale_VSA', 'invalid_CAN', 'pose', 'gas_pedal', 'brake_pedal', 'inactive', 'not_pid',
                                    'low_speed', 'high_speed', 'no_offset', 'new_domain', 'bridge', 'received_brake', 'pressure', 'forecast'),
                                   (False, True)):
      with self.subTest(change=change, eco_mode=eco_mode):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., .08, 0.])
        control.actuators.accel = .05
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=15., aEgo=.05, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=0, odyssey_target_gear=4,
                             odyssey_user_brake=0., odyssey_computer_braking=False)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)

        for frame in range(200 if change in (None, 'at_target', 'gear', 'activity') else 142):
          now = 2_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
          if frame == 60:
            control.actuators.accel, state.aEgo = -.25, -.34
          if frame == 140:
            self.assertFalse(controller.odyssey_coast_response.drag)
            self.assertGreater(controller.odyssey_gas_response.correction, 0.)
            self.assertEqual(len(controller.odyssey_gas_response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
            control.actuators.accel = -.27
            if change == 'at_target':
              state.aEgo = control.actuators.accel
            elif change == 'excess':
              state.aEgo = .1
            elif change == 'stronger':
              control.actuators.accel = -.95
            elif change == 'more_decel':
              control.actuators.accel = -.4
            elif change == 'gear':
              CS.odyssey_target_gear = 5
            elif change == 'activity':
              CS.odyssey_shift_activity = 1
            elif change == 'stale_engine':
              CS.odyssey_engine_torque_ts_nanos -= 80_000_000
            elif change == 'stale_gear':
              CS.odyssey_target_gear_ts_nanos -= 80_000_000
            elif change == 'stale_VSA':
              CS.odyssey_computer_braking_ts_nanos -= 80_000_000
            elif change == 'invalid_CAN':
              state.canValid = False
            elif change == 'pose':
              control.orientationNED = []
            elif change == 'gas_pedal':
              state.gasPressed = True
            elif change == 'brake_pedal':
              state.brakePressed = True
            elif change == 'inactive':
              control.longActive = False
            elif change == 'not_pid':
              control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
            elif change == 'low_speed':
              state.vEgo = 7.9
            elif change == 'high_speed':
              state.vEgo = 35.1
            elif change == 'no_offset':
              controller.odyssey_gas_response.correction = 0.
            elif change == 'new_domain':
              controller.odyssey_gas_selected = False
            elif change == 'bridge':
              controller.odyssey_gas_bridge_active = True
            elif change == 'received_brake':
              CS.odyssey_computer_braking = True
            elif change == 'pressure':
              CS.odyssey_user_brake = .03125
            elif change == 'forecast':
              controller.odyssey_coast_response.drag[4] = -.1 + ACCELERATION_DUE_TO_GRAVITY * math.sin(.08)
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          messages = [(addr, dat, bus) for addr, dat, bus in sends if addr == 0x1DF and bus == 1]
          if not messages:
            continue
          for addr, dat, bus in messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))
          parser.update([(now, messages)])
          values = parser.vl['ACC_CONTROL']
          if frame >= 140:
            self.assertAlmostEqual(values['ACCEL_COMMAND'], control.actuators.accel if control.longActive else 0., delta=.01)
            self.assertAlmostEqual(output.accel, values['ACCEL_COMMAND'], delta=.01)
            self.assertEqual(values['BRAKE_REQUEST'], change == 'stronger')
            if change in (None, 'at_target', 'more_decel', 'gear', 'activity'):
              self.assertGreater(values['GAS_COMMAND'], 0)
              if change in ('gear', 'activity'):
                # Restart delayed history while retaining gas still needed by the live response.
                self.assertEqual(controller.odyssey_gas_response.drivetrain, (CS.odyssey_target_gear, CS.odyssey_shift_activity))
                self.assertEqual(len(controller.odyssey_gas_response.requests),
                                 min((frame - 140) // 2 + 1, ODYSSEY_RESPONSE_DELAY_FRAMES + 1))
              else:
                self.assertEqual(len(controller.odyssey_gas_response.requests), ODYSSEY_RESPONSE_DELAY_FRAMES + 1)
            else:
              self.assertEqual(values['GAS_COMMAND'], -30000)

  def test_prior_coast_response_braking_cancels_gas_preactivation(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    controller.frame = 100
    controller.odyssey_coast_response.gear = 7
    controller.odyssey_coast_response.coast_start = 98
    controller.odyssey_coast_response.drag[7] = 0.3
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, 0.0, 0.0]
    control.actuators.accel = -0.05
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.aEgo, state.canValid = 22.0, 0.3, True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=1_990_000_000, odyssey_shift_activity=0, odyssey_target_gear=7,
                         odyssey_target_gear_ts_nanos=1_990_000_000,
                         odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=1_990_000_000)
    _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, 2_000_000_000)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    parser.update([(2_000_000_000, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
    self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
    self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], 1)
    self.assertFalse(controller.odyssey_gas_bridge_active)

  def test_settled_coast_deceleration_shortfall_enters_brakes_and_releases_above_coast(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    control = structs.CarControl()
    control.enabled = control.longActive = True
    control.orientationNED = [0.0, -0.03, 0.0]
    control.actuators.accel = -0.15
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState()
    state.vEgo, state.aEgo, state.canValid = 20.0, 0.3, True
    CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0.0,
                         odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                         odyssey_target_gear_ts_nanos=0,
                         odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)

    def step():
      now = 1_000_000_000 + controller.frame * 10_000_000
      CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
      output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      parser.update([(now, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
      return output, parser.vl['ACC_CONTROL']

    for _ in range(50):
      _, signals = step()
      self.assertEqual(signals['BRAKE_REQUEST'], 0)
    for _ in range(50):
      output, signals = step()
    self.assertEqual(signals['BRAKE_REQUEST'], 1)
    self.assertEqual(signals['GAS_COMMAND'], -30000)
    self.assertAlmostEqual(signals['ACCEL_COMMAND'], -0.15, delta=0.01)
    self.assertAlmostEqual(output.accel, signals['ACCEL_COMMAND'], delta=0.01)
    control.actuators.accel = 0.4
    state.aEgo = 0.0
    _, signals = step()
    self.assertEqual(signals['BRAKE_REQUEST'], 0)
    self.assertGreater(signals['GAS_COMMAND'], 0)

  def test_cold_coast_shortfall_enters_light_brake_and_retains_its_release_forecast(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    for request in (-.20, 0., .1):
      with self.subTest(request=request):
        CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        dbc = DBC[CP.carFingerprint][Bus.pt]
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0.0, -0.03, 0.0])
        control.actuators.accel = min(request, 0.)
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=20.0, aEgo=0.35, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0.0,
                             odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_target_gear_ts_nanos=0,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(80):
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          if frame == 60:
            control.actuators.accel = request
          elif frame == 62:
            control.actuators.accel = -0.13
            state.aEgo = -0.5
            CS.odyssey_computer_braking = True
          elif frame == 74:
            control.actuators.accel = 0.1
          elif frame == 76:
            control.actuators.accel = 0.4
          elif frame == 78:
            CS.odyssey_computer_braking = False
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          if frame % 2:
            continue
          messages = [s for s in sends if s[0] == 0x1DF and s[2] == 1]
          for addr, dat, bus in messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))
          parser.update([(now, messages)])
          signals = parser.vl['ACC_CONTROL']
          self.assertEqual(signals['BRAKE_REQUEST'], 60 <= frame < 76)
          if 60 <= frame < 76:
            self.assertEqual(signals['GAS_COMMAND'], -30000)
            self.assertAlmostEqual(signals['ACCEL_COMMAND'], control.actuators.accel, delta=0.01)
            self.assertAlmostEqual(output.accel, signals['ACCEL_COMMAND'], delta=0.01)
            self.assertIsNotNone(controller.odyssey_coast_response.drag.get(7))
          if frame == 76:
            self.assertEqual(signals['GAS_COMMAND'], -30000)
          if frame == 78:
            self.assertGreater(signals['GAS_COMMAND'], 0)

  def test_bad_coast_inputs_cannot_authorize_braking_after_reception_recovers(self):
    from opendbc.safety.tests.libsafety import libsafety_py

    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    faults = ('invalid_can', 'missing_gear', 'stale_gear', 'future_gear', 'missing_brake', 'stale_brake',
              'future_brake', 'missing_gas', 'stale_gas', 'future_gas')
    for blocked, known in [(None, False)] + [(fault, known) for fault in faults for known in (False, True)]:
      with self.subTest(blocked=blocked, known=known):
        safety = libsafety_py.libsafety
        config = CP.safetyConfigs[-1]
        self.assertEqual(safety.set_safety_hooks(config.safetyModel.raw, config.safetyParam), 0)
        safety.init_tests()
        safety.set_controls_allowed(True)
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        if known:
          for frame in range(0, 64, 2):
            controller.odyssey_coast_response.update(frame, 7, -0.03, 0.3, True, True)
          controller.frame = 64
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0.0, -0.03, 0.0])
        control.actuators.accel = -0.15
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=20.0, aEgo=0.3, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0.0,
                             odyssey_engine_torque_ts_nanos=0, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_target_gear_ts_nanos=0,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=0)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(144):
          now = 1_000_000_000 + frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
          state.canValid = True
          if blocked and frame <= 80:
            if blocked == 'invalid_can':
              state.canValid = False
            else:
              fault, signal = blocked.split('_')
              timestamp = {'missing': 0, 'stale': now - 60_000_000, 'future': now + 1}[fault]
              name = {'gear': 'odyssey_target_gear_ts_nanos', 'brake': 'odyssey_computer_braking_ts_nanos',
                      'gas': 'odyssey_engine_torque_ts_nanos'}[signal]
              setattr(CS, name, timestamp)
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          if frame % 2:
            continue
          messages = [s for s in sends if s[0] == 0x1DF and s[2] == 1]
          for addr, dat, bus in messages:
            self.assertTrue(safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat)))
          parser.update([(now, messages)])
          first_brake = 60 if blocked is None else 142
          self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], frame >= first_brake)
          self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)

  def test_coast_brake_entry_requires_fresh_reliable_excess_acceleration(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for request, blocked in product((-.15, 0., .1), (None, 'short_coast', 'no_history', 'stale', 'pose', 'gas_pedal', 'brake_pedal', 'inactive',
                    'not_pid', 'low_speed', 'high_speed', 'unknown_gear', 'forecast', 'response',
                    'invalid_can', 'stale_gear', 'missing_gear', 'future_gear', 'stale_brake', 'missing_brake',
                    'future_brake', 'received_gas', 'received_brake', 'previous_gas')):
      with self.subTest(request=request, blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_pitch.x = -0.03
        controller.odyssey_coast_response.gear = 7
        controller.odyssey_coast_response.coast_start = 0
        controller.odyssey_coast_response.drag[7] = 0.5 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.03)
        control = structs.CarControl()
        control.enabled = control.longActive = True
        control.orientationNED = [0.0, -0.03, 0.0]
        control.actuators.accel = request
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState()
        state.vEgo, state.aEgo, state.canValid = 20.0, 0.5, True
        CS = SimpleNamespace(out=state, v_cruise_factor=0.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0.0,
                             odyssey_engine_torque_ts_nanos=1_000_000_000, odyssey_shift_activity=0, odyssey_target_gear=7,
                             odyssey_target_gear_ts_nanos=1_000_000_000,
                             odyssey_user_brake=0., odyssey_computer_braking=False, odyssey_computer_braking_ts_nanos=1_000_000_000)
        if blocked == 'short_coast':
          controller.odyssey_coast_response.coast_start = 98
        elif blocked == 'no_history':
          controller.odyssey_coast_response.drag.clear()
        elif blocked == 'stale':
          CS.odyssey_engine_torque_ts_nanos -= 80_000_000
        elif blocked == 'pose':
          control.orientationNED = []
        elif blocked == 'gas_pedal':
          state.gasPressed = True
        elif blocked == 'brake_pedal':
          state.brakePressed = True
        elif blocked == 'inactive':
          control.longActive = False
        elif blocked == 'not_pid':
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.stopping
        elif blocked == 'low_speed':
          state.vEgo = 4.0
        elif blocked == 'high_speed':
          state.vEgo = 36.0
        elif blocked == 'unknown_gear':
          CS.odyssey_target_gear = 8
        elif blocked == 'forecast':
          controller.odyssey_coast_response.drag[7] = -0.1 + ACCELERATION_DUE_TO_GRAVITY * math.sin(-0.03)
        elif blocked == 'response':
          state.aEgo = -0.1
        elif blocked == 'invalid_can':
          state.canValid = False
        elif blocked in ('stale_gear', 'missing_gear', 'future_gear'):
          CS.odyssey_target_gear_ts_nanos = {
            'stale_gear': 920_000_000, 'missing_gear': 0, 'future_gear': 1_000_000_001,
          }[blocked]
        elif blocked in ('stale_brake', 'missing_brake', 'future_brake'):
          CS.odyssey_computer_braking_ts_nanos = {
            'stale_brake': 920_000_000, 'missing_brake': 0, 'future_brake': 1_000_000_001,
          }[blocked]
        elif blocked == 'received_gas':
          CS.odyssey_car_gas = 1.0
        elif blocked == 'received_brake':
          CS.odyssey_computer_braking = True
        elif blocked == 'previous_gas':
          controller.odyssey_gas_selected = True
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, 1_000_000_000)
        parser.update([(1_000_000_000, [(addr, dat, src) for addr, dat, src in sends if addr == 0x1DF and src == 1])])
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'],
                         blocked in (None, 'short_coast') or (blocked == 'low_speed' and request <= 0.0))
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], request if blocked != 'inactive' else 0., delta=.01)

  def test_brake_feedback_release_requires_fresh_state_and_excess_deceleration(self):
    for case in (None, 'coast', 'strong', 'stale', 'future', 'missing', 'gas', 'gear', 'ineligible',
                 'response', 'positive', 'zero', 'forecast', 'invalid_forecast'):
      with self.subTest(case=case):
        now = 1_000_000_000
        request = -.4 if case == 'strong' else .1 if case == 'positive' else 0. if case == 'zero' else -.2
        timestamp = {'stale': now - 60_000_000, 'future': now + 1, 'missing': 0}.get(case, now - 5_000_000)
        passive = {'forecast': request + .01, 'invalid_forecast': -math.inf}.get(case, request - .01)
        release = odyssey_brake_release(now, request, request - (.1 if case == 'response' else .3),
                                        case != 'coast', timestamp, 1. if case == 'gas' else 0.,
                                        0 if case == 'gear' else 6, case != 'ineligible', passive)
        self.assertEqual(release, case in (None, 'coast', 'strong'))
        if release:
          self.assertEqual(odyssey_command_domains(request, 20., previous_brake=True, release_brake=release), (False, False))
          self.assertEqual(odyssey_command_domains(request, 4., previous_brake=True, release_brake=release), (False, True))

  def test_brake_release_retains_coast_until_request_or_response_needs_braking(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
    controller.frame = 100
    controller.odyssey_brake_selected = True
    controller.odyssey_brake_start_frame = 0
    controller.odyssey_pitch.x = .04
    controller.odyssey_coast_response.gear = 7
    drag = -.5 + ACCELERATION_DUE_TO_GRAVITY * math.sin(.04)
    controller.odyssey_coast_response.drag[7] = drag
    control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., .04, 0.])
    control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
    state = structs.CarState(vEgo=22., canValid=True)
    CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                         odyssey_car_gas=0., odyssey_shift_activity=0, odyssey_target_gear=7,
                         odyssey_user_brake=.125, odyssey_computer_braking=True)
    parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
    # Brake/coast hysteresis uses the existing response margin; numeric requests remain intact.
    for request, actual, received_brake, expected_brake in [(-.4, -.5, True, True),
                                                           (-.4, -.7, True, False),
                                                           (-.4, -.4, True, False),
                                                           (-.4, -.4, False, False),
                                                           (-.4, -.21, False, False),
                                                           (-.4, -.19, False, True),
                                                           (-.4, -.7, True, False),
                                                           (-.6, -.6, False, True),
                                                           (0., -.4, True, False)]:
      control.actuators.accel = request
      state.aEgo = actual
      CS.odyssey_computer_braking = received_brake
      for _ in range(2):
        now = 2_000_000_000 + controller.frame * 10_000_000
        CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
        output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser.update([(now, [(addr, dat, bus) for addr, dat, bus in sends if addr == 0x1DF and bus == 1])])
      values = parser.vl['ACC_CONTROL']
      self.assertEqual(values['BRAKE_REQUEST'], expected_brake)
      self.assertEqual(values['GAS_COMMAND'], -30000)
      self.assertAlmostEqual(values['ACCEL_COMMAND'], request, delta=.01)
      self.assertAlmostEqual(output.accel, request, delta=.00001)
      self.assertEqual(controller.odyssey_coast_response.drag[7], drag)
      self.assertFalse(controller.odyssey_coast_response.warmup)

  def test_brake_release_does_not_reverse_on_a_small_forecast_shortfall_while_overbraking(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for econ in (False, True):
      with self.subTest(econ=econ):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        controller.odyssey_pitch.x = .02
        controller.odyssey_coast_response.gear = 6
        drag = -.36 + ACCELERATION_DUE_TO_GRAVITY * math.sin(.02)
        controller.odyssey_coast_response.drag[6] = drag
        control = structs.CarControl(enabled=True, longActive=True)
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=15., canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=102, odyssey_target_gear=6,
                             odyssey_user_brake=.25, odyssey_computer_braking=True)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for request, actual, forecast, braking in [(-.35, -.70, -.36, False),
                                                    (-.35, -.68, -.34, False),
                                                    (-.45, -.45, -.34, True),
                                                    (-.35, -.70, -.36, False),
                                                    (-.35, -.34, -.34, True),
                                                    (-.35, -.70, -.36, False),
                                                    (-.35, -.70, -.10, True),
                                                    (-.35, -.70, -.34, True)]:
          control.actuators.accel, state.aEgo = request, actual
          control.orientationNED = [0., math.asin((drag - forecast) / ACCELERATION_DUE_TO_GRAVITY), 0.]
          for _ in range(2):
            now = 1_000_000_000 + controller.frame * 10_000_000
            CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
            output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
            parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          values = parser.vl['ACC_CONTROL']
          self.assertEqual(values['BRAKE_REQUEST'], braking)
          self.assertEqual(values['GAS_COMMAND'], -30000)
          self.assertAlmostEqual(values['ACCEL_COMMAND'], request, delta=.01)
          self.assertAlmostEqual(output.accel, request, delta=.00001)
          self.assertEqual(controller.odyssey_coast_response.drag[6], drag)
          self.assertFalse(controller.odyssey_coast_response.warmup)

  def test_settled_excess_braking_can_release_to_a_close_coast_forecast(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for econ in (False, True):
      with self.subTest(econ=econ):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.accel = -.2
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        controller.odyssey_coast_response.gear = 6
        controller.odyssey_coast_response.drag[6] = -.15
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = -.2
        controller.accel = control.actuators.accel
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=20., aEgo=-.55, canValid=True)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=102, odyssey_target_gear=6,
                             odyssey_user_brake=.25, odyssey_computer_braking=True)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        released = False
        for _ in range(50):
          now = 1_000_000_000 + controller.frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
          output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          if not any(msg[0] == 0x1DF for msg in sends):
            continue
          parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
          values = parser.vl['ACC_CONTROL']
          if not released:
            self.assertEqual(values['BRAKE_REQUEST'], controller.odyssey_brake_response_error.x <= .2)
          released |= not values['BRAKE_REQUEST']
          self.assertEqual(values['GAS_COMMAND'], -30000)
          self.assertAlmostEqual(values['ACCEL_COMMAND'], -.2, delta=.01)
          self.assertAlmostEqual(output.accel, -.2, delta=1e-5)
        self.assertTrue(released)
        self.assertEqual(controller.odyssey_coast_response.drag[6], -.15)
        self.assertFalse(controller.odyssey_coast_response.warmup)
        control.actuators.accel = -.35
        for _ in range(2):
          now = 2_000_000_000 + controller.frame * 10_000_000
          CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        self.assertTrue(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'])
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], -.35, delta=.01)

  def test_initial_coast_margin_requires_settled_consistent_excess(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for econ, blocked in product((False, True), (None, 'cold', 'unsettled', 'gear_change', 'stronger',
                                               'response', 'forecast', 'invalid_can', 'low_speed', 'high_speed')):
      with self.subTest(econ=econ, blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.accel = -.1 if blocked == 'stronger' else -.2
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 80 if blocked == 'unsettled' else 0
        controller.odyssey_brake_response_error.x = 0. if blocked == 'cold' else .3
        controller.odyssey_coast_response.gear = 5 if blocked == 'gear_change' else 6
        controller.odyssey_coast_response.drag[6] = .01 if blocked == 'forecast' else -.15
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = -.2
        if blocked != 'stronger':
          controller.accel = control.actuators.accel
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=7.9 if blocked == 'low_speed' else 35.1 if blocked == 'high_speed' else 20.,
                                 aEgo=-.35 if blocked == 'response' else -.55, canValid=blocked != 'invalid_can')
        now = 2_000_000_000
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=102, odyssey_target_gear=6,
                             odyssey_user_brake=.25, odyssey_computer_braking=True,
                             odyssey_engine_torque_ts_nanos=now, odyssey_target_gear_ts_nanos=now,
                             odyssey_computer_braking_ts_nanos=now)
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], blocked is not None)
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], -.2, delta=.01)

  def test_forecast_tolerance_requires_a_settled_same_gear_brake_handoff(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for start, previous_gear, expected_brake in [(None, 6, True), (80, 6, True), (0, 5, True), (0, 6, False)]:
      with self.subTest(start=start, previous_gear=previous_gear):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.accel = -.35
        controller.odyssey_brake_start_frame = start
        controller.odyssey_coast_response.gear = previous_gear
        controller.odyssey_coast_response.drag[6] = -.34
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = -.35
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=15., aEgo=-.65, canValid=True)
        now = 2_000_000_000
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=102, odyssey_target_gear=6,
                             odyssey_user_brake=.25, odyssey_computer_braking=True,
                             odyssey_engine_torque_ts_nanos=now, odyssey_target_gear_ts_nanos=now,
                             odyssey_computer_braking_ts_nanos=now)
        _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], expected_brake)
        self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], -.35, delta=.01)

  def test_initial_forecast_margin_requires_a_qualified_handoff(self):
    for context, forecast in product((False, True), (-.34, -.16, -.14)):
      with self.subTest(context=context, forecast=forecast):
        self.assertEqual(odyssey_brake_release(1_000_000_000, -.35, -.7, True, 995_000_000,
                                               0., 6, True, forecast, context), context and forecast <= -.15)

  def test_forecast_tolerance_preserves_needed_braking(self):
    for forecast, actual, released in [(-.34, -.65, True), (-.15, -.65, True), (-.14, -.65, False),
                                        (-.34, -.34, False), (-.34, -.35, True), (-.55, -.12, False)]:
      with self.subTest(forecast=forecast, actual=actual):
        self.assertEqual(odyssey_brake_release(1_000_000_000, -.35, actual, False, 995_000_000,
                                               0., 6, True, forecast, True), released)

  def test_brake_release_recognizes_feedback_when_computer_braking_is_idle(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for blocked in (None, 'one_step', 'zero', 'negative', 'nan', 'infinite', 'computer_only',
                    'stale', 'future', 'invalid_can', 'unlearned', 'weak_coast', 'response', 'low_speed', 'brake_pedal'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        controller.frame = 100
        controller.odyssey_brake_selected = True
        controller.odyssey_brake_start_frame = 0
        controller.odyssey_coast_response.gear = 7
        if blocked != 'unlearned':
          controller.odyssey_coast_response.drag[7] = -.15 if blocked == 'weak_coast' else -.25
        control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., 0., 0.])
        control.actuators.accel = -.2
        control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
        state = structs.CarState(vEgo=7.5 if blocked == 'low_speed' else 20.,
                                 aEgo=-.3 if blocked == 'response' else -.45,
                                 canValid=blocked != 'invalid_can', brakePressed=blocked == 'brake_pedal')
        now = 2_000_000_000
        pressure = {'one_step': .015625, 'zero': 0., 'negative': -.125, 'nan': math.nan,
                    'infinite': math.inf, 'computer_only': math.nan}.get(blocked, .0625)
        CS = SimpleNamespace(out=state, v_cruise_factor=.44704, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={},
                             odyssey_car_gas=0., odyssey_shift_activity=119, odyssey_target_gear=7,
                             odyssey_engine_torque_ts_nanos=now, odyssey_target_gear_ts_nanos=now,
                             odyssey_computer_braking=blocked == 'computer_only', odyssey_user_brake=pressure,
                             odyssey_computer_braking_ts_nanos={'stale': now - 60_000_000, 'future': now + 1}.get(blocked, now))
        parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
        parser.update([(now, [(addr, dat, bus) for addr, dat, bus in sends if addr == 0x1DF and bus == 1])])
        values = parser.vl['ACC_CONTROL']
        self.assertEqual(values['BRAKE_REQUEST'], blocked not in (None, 'computer_only'))
        self.assertEqual(values['GAS_COMMAND'], -30000)
        self.assertAlmostEqual(values['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
        self.assertAlmostEqual(output.accel, control.actuators.accel, delta=.00001)
        self.assertFalse(controller.odyssey_coast_response.warmup)

  def test_brake_release_requires_coast_to_supply_requested_deceleration(self):
    for coast_error in (None, -.01, 0., .001, .1, .19):
      with self.subTest(coast_error=coast_error):
        request = -.2
        passive_accel = None if coast_error is None else request + coast_error
        release = odyssey_brake_release(1_000_000_000, request, request - .3, True, 995_000_000,
                                        0., 6, True, passive_accel)
        self.assertEqual(release, coast_error is not None and coast_error <= 0.)

  def test_brake_release_uses_received_braking_with_flat_engine_torque(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    packer = CANPacker(dbc)
    for blocked in (None, 'no_brake', 'stale_brake', 'stale_gas', 'invalid_can', 'gas_pedal',
                    'brake_pedal', 'stopping', 'low_speed', 'strong', 'strong_coast', 'coast_shortfall', 'coast_weak',
                    'coast_sufficient', 'stale_gear', 'missing_gear', 'future_gear', 'invalid_pose'):
      with self.subTest(blocked=blocked):
        controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
        CS = CarState(CP, _sp(CP))
        parsers = CS.get_can_parsers(CP, _sp(CP))
        CS.update(parsers)
        CS.out = structs.CarState(vEgo=7.5 if blocked == 'low_speed' else 20., canValid=blocked != 'invalid_can',
                                  gasPressed=blocked == 'gas_pedal', brakePressed=blocked == 'brake_pedal')
        control = structs.CarControl(longActive=True, enabled=True)
        forecast_case = blocked in ('strong_coast', 'coast_shortfall', 'coast_weak', 'coast_sufficient', 'stale_gear', 'missing_gear',
                                   'future_gear', 'invalid_pose')
        if forecast_case:
          control.orientationNED = [] if blocked == 'invalid_pose' else [0., -.04, 0.]
          controller.odyssey_pitch.x = -.04
          controller.odyssey_coast_response.gear = 6
          passive_accel = {'strong_coast': -.5, 'coast_sufficient': -.5, 'coast_weak': 0.}.get(blocked, .3)
          controller.odyssey_coast_response.drag[6] = passive_accel + ACCELERATION_DUE_TO_GRAVITY * math.sin(-.04)
        control.actuators.longControlState = (structs.CarControl.Actuators.LongControlState.stopping if blocked == 'stopping'
                                              else structs.CarControl.Actuators.LongControlState.pid)
        output_parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
        for frame in range(24):
          now = 1_000_000_000 + frame * 10_000_000
          signals = [('VSA_STATUS', {'COMPUTER_BRAKING': blocked != 'no_brake'}),
                     ('GAS_PEDAL_2', {'ENGINE_TORQUE_ESTIMATE': -120, 'CAR_GAS': 0}),
                     ('GEARBOX_AUTO', {'TRANS_TARGET_GEAR': 6})]
          for name, values in signals:
            if frame > 0 and ((blocked == 'stale_brake' and name == 'VSA_STATUS') or
                              (blocked == 'stale_gas' and name == 'GAS_PEDAL_2') or
                              (blocked == 'stale_gear' and name == 'GEARBOX_AUTO')):
              continue
            parsers[Bus.pt].update([(now - 5_000_000, [packer.make_can_msg(name, 1, values)])])
          CS.update(parsers)
          if blocked in ('missing_gear', 'future_gear'):
            CS.odyssey_target_gear_ts_nanos = 0 if blocked == 'missing_gear' else now + 1
          control.actuators.accel = -.4 if blocked in ('strong', 'strong_coast') else -.28 + frame * .004
          CS.out.aEgo = control.actuators.accel - .3
          if frame == 0:
            controller.odyssey_brake_selected = True
          _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
          output_parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
        retain_brake = blocked not in ('coast_sufficient', 'strong_coast')
        self.assertEqual(output_parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], retain_brake)
        self.assertEqual(output_parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)
        if blocked is None:
          self.assertAlmostEqual(output_parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], control.actuators.accel, delta=.01)
        if blocked in ('coast_shortfall', 'coast_weak'):
          for request in (0., .1, .4):
            control.actuators.accel = request
            for _ in range(2):
              now += 10_000_000
              CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
              _, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
              output_parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
            self.assertEqual(output_parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], blocked == 'coast_shortfall' and request < .3)

  def test_uphill_negative_gas_demand_delays_only_an_active_gas_release(self):
    gas_accel = odyssey_grade_gas_accel(-0.23, 0.03)
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True, gas_accel=gas_accel), (True, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, gas_accel=gas_accel), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True, gas_accel=-0.23), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True,
                                              gas_accel=odyssey_grade_gas_accel(-0.23, 0.01)), (False, False))
    self.assertEqual(odyssey_command_domains(-0.23, 20.0, previous_gas=True,
                                              gas_accel=odyssey_grade_gas_accel(-0.23, 0.07)), (True, False))

  def test_road_brake_wire_preserves_request_across_grade_and_response(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    for pitch in (-.08, -.03, 0., .03, .08, None, math.nan):
      for actual in (-2., 2.):
        with self.subTest(pitch=pitch, actual=actual):
          controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
          CS = CarState(CP, _sp(CP))
          CS.update(CS.get_can_parsers(CP, _sp(CP)))
          CS.out = structs.CarState(vEgo=20., aEgo=actual, canValid=True)
          control = structs.CarControl(enabled=True, longActive=True,
                                       orientationNED=[] if pitch is None else [0., pitch, 0.])
          control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
          parser = CANParser(dbc, [('ACC_CONTROL', 0)], 1)
          for request in (-4., -.5, -.15, -.05, -.001, 0., .1, 3.):
            control.actuators.accel = request
            for _ in range(4):
              now = 1_000_000_000 + controller.frame * 10_000_000
              output, sends = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
              parser.update([(now, [msg for msg in sends if msg[0] == 0x1DF])])
            expected = min(max(request, CarControllerParams.BOSCH_ACCEL_MIN), CarControllerParams.BOSCH_ACCEL_MAX)
            self.assertAlmostEqual(parser.vl['ACC_CONTROL']['ACCEL_COMMAND'], expected, delta=.01)
            self.assertAlmostEqual(output.accel, expected, delta=.00001)
            self.assertEqual(parser.vl['ACC_CONTROL']['BRAKE_REQUEST'], request < 0.)
            if request < 0.:
              self.assertEqual(parser.vl['ACC_CONTROL']['GAS_COMMAND'], -30000)

  def test_negative_gas_bridge_preserves_existing_domain_lifecycle(self):
    gas, active = odyssey_gas_command(-0.10, 91.0, True, False, False)
    self.assertEqual((gas, active), (ODYSSEY_GAS_BRIDGE_COMMAND, True))
    gas, active = odyssey_gas_command(-0.05, 136.0, True, True, active)
    self.assertEqual((gas, active), (ODYSSEY_GAS_BRIDGE_COMMAND, True))
    gas, active = odyssey_gas_command(0.0, 182.0, True, True, active)
    self.assertEqual((gas, active), (182.0, False))
    gas, active = odyssey_gas_command(-0.10, 91.0, True, True, False)
    self.assertEqual((gas, active), (91.0, False))
    gas, active = odyssey_gas_command(-0.15, 45.0, False, True, True)
    self.assertEqual((gas, active), (0.0, False))
    self.assertEqual(odyssey_gas_command(-0.05, 136.0, True, False, False, 0.5), (38.0, True))
    self.assertEqual(odyssey_gas_command(-0.05, 136.0, True, False, False, 1.0), (136.0, True))
    self.assertEqual(odyssey_gas_command(0.0, 182.0, True, True, True, 1.0), (182.0, False))

  def test_actuator_delay_matches_controller_tuning(self):
    # the Odyssey controller was tuned and driven with the stock 0.5 s planner delay; other Bosch cars keep 0.25 s
    for alpha_long in (False, True):
      CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], alpha_long, False, False)
      self.assertAlmostEqual(CP.longitudinalActuatorDelay, 0.5)
    CP = CarInterface.get_params(CAR.HONDA_ACCORD, gen_empty_fingerprint(), [], True, False, False)
    self.assertAlmostEqual(CP.longitudinalActuatorDelay, 0.25)

  def test_load_preset_maps_to_bounded_gas_scale(self):
    self.assertEqual([odyssey_load_factor(p) for p in (0, 1, 2)], list(ODYSSEY_LOAD_FACTORS))
    for bad in (None, -1, 3, "x", 1.5e9):
      self.assertEqual(odyssey_load_factor(bad), 1.0)
    self.assertEqual(ODYSSEY_LOAD_FACTORS[0], 1.0)
    self.assertTrue(all(1.0 <= f <= 1.3 for f in ODYSSEY_LOAD_FACTORS))
    self.assertEqual(sorted(ODYSSEY_LOAD_FACTORS), list(ODYSSEY_LOAD_FACTORS))

  def test_load_scales_only_positive_gas_map_input(self):
    for factor in ODYSSEY_LOAD_FACTORS:
      self.assertAlmostEqual(odyssey_load_gas_accel(0.8, factor), 0.8 * factor)
      self.assertEqual(odyssey_load_gas_accel(0.0, factor), 0.0)
      self.assertEqual(odyssey_load_gas_accel(-0.15, factor), -0.15)

  def test_load_preset_raises_issued_gas_within_map_limit(self):
    CP = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, gen_empty_fingerprint(), [], True, False, False)
    dbc = DBC[CP.carFingerprint][Bus.pt]
    gas_max = CarControllerParams.BOSCH_GAS_LOOKUP_V[-1]

    def issued_gas(factor, accel, pitch=0.):
      with patch.object(CarController, "_read_odyssey_load_factor", lambda self: factor):
        return _issued_gas(accel, pitch)

    def _issued_gas(accel, pitch):
      controller = CarController({Bus.pt: dbc}, CP, _sp(CP))
      CS = CarState(CP, _sp(CP))
      CS.update(CS.get_can_parsers(CP, CS.CP_SP))
      CS.out = structs.CarState(vEgo=20., aEgo=accel, canValid=True)
      CS.odyssey_target_gear = 7
      CS.odyssey_car_gas = 0.
      CS.odyssey_computer_braking = False
      CS.odyssey_user_brake = 0.
      control = structs.CarControl(enabled=True, longActive=True, orientationNED=[0., pitch, 0.])
      control.actuators.accel = accel
      control.actuators.longControlState = structs.CarControl.Actuators.LongControlState.pid
      output = None
      for _ in range(4):
        now = 1_000_000_000 + controller.frame * 10_000_000
        CS.odyssey_engine_torque_ts_nanos = CS.odyssey_target_gear_ts_nanos = CS.odyssey_computer_braking_ts_nanos = now - 5_000_000
        output, _ = controller.update(control.as_reader(), structs.CarControlSP(), CS, now)
      return output.gas

    for accel, pitch in ((0.5, 0.), (1.0, 0.), (0.3, 0.04)):
      empty = issued_gas(1.0, accel, pitch)
      self.assertGreater(empty, 0.)
      for factor in ODYSSEY_LOAD_FACTORS[1:]:
        loaded = issued_gas(factor, accel, pitch)
        self.assertGreater(loaded, empty)
        # the extra gas is the load share of the mapped request, never more than that share of the map
        self.assertLessEqual(loaded, gas_max)
        self.assertLessEqual(loaded - empty, (factor - 1.) * gas_max + 1.)
    # a full request already at the map limit stays there under any load
    for factor in ODYSSEY_LOAD_FACTORS:
      self.assertLessEqual(issued_gas(factor, 2.0), gas_max)
