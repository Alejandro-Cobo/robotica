from pyrobot.brain import Brain

import math

import cv2
import numpy as np

import lib

class BrainTestNavigator(Brain):

  def setup(self):
    # TODO: create video capture, create and train classifiers.
    pass

  def step(self):
    # TODO: segment each frame of the video.
    pass
 
def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  # If we are allowed (for example you can't in a simulation), enable
  # the motors.
  try:
    engine.robot.position[0]._dev.enable(1)
  except AttributeError:
    pass

  return BrainTestNavigator('BrainTestNavigator', engine)
