import pytest

from opendbc.can.parser import CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC


@pytest.fixture
def odyssey_hud():
  fingerprint = gen_empty_fingerprint()
  cp = CarInterface.get_params(CAR.HONDA_ODYSSEY_5G_MMR, fingerprint, [], True, False, False)
  sp = CarInterface.get_params_sp(cp, cp.carFingerprint, fingerprint, [], True, False, False)
  state = CarState(cp, sp)
  state.out, state.out_sp = state.update(state.get_can_parsers(cp, sp))
  state.out.vEgo = 22.0
  state.out.cruiseState.available = True
  state.v_cruise_factor = CV.MPH_TO_MS
  state.is_metric = False
  dbc = DBC[cp.carFingerprint][Bus.pt]
  controller = CarController({Bus.pt: dbc}, cp, sp)
  parser = CANParser(dbc, [('ACC_HUD', 0), ('LKAS_HUD', 0)], controller.CAN.pt)
  return controller, state, parser


@pytest.mark.parametrize('engaged,cruise_speed,expected', [
  (False, 100.0, 65),
  (True, 100.0, 65),
  (False, 0.0, 255),
  (False, 255.0, 255),
])
def test_cruise_number_remains_visible_without_enabling_acc(odyssey_hud, engaged, cruise_speed, expected):
  controller, state, parser = odyssey_hud
  state.out.vCruise = cruise_speed
  control = structs.CarControl()
  control.enabled = engaged
  control.hudControl.speedVisible = engaged
  control.hudControl.setSpeed = 65 * CV.MPH_TO_MS
  _, messages = controller.update(control.as_reader(), structs.CarControlSP(), state, 0)
  parser.update([(0, messages)])
  assert parser.vl['ACC_HUD']['CRUISE_SPEED'] == expected
  assert parser.vl['ACC_HUD']['ACC_ON'] == int(engaged)


@pytest.mark.parametrize('available,armed,lateral,visible,dashed,solid', [
  (True, False, False, False, 0, 0),
  (True, True, False, False, 1, 0),
  (True, True, True, False, 1, 1),
  (False, False, False, True, 1, 0),
])
def test_lane_indicators_follow_mads_even_with_acc_off(odyssey_hud, available, armed, lateral, visible, dashed, solid):
  controller, state, parser = odyssey_hud
  control = structs.CarControl()
  control.latActive = lateral
  control.hudControl.lanesVisible = visible
  control_sp = structs.CarControlSP()
  control_sp.mads.available = available
  control_sp.mads.enabled = armed
  _, messages = controller.update(control.as_reader(), control_sp, state, 0)
  parser.update([(0, messages)])
  hud = parser.vl['LKAS_HUD']
  assert (hud['DASHED_LANES'], hud['SOLID_LANES']) == (dashed, solid)
  assert hud['RDM_OFF'] == 1
  assert parser.vl['ACC_HUD']['ACC_ON'] == 0
