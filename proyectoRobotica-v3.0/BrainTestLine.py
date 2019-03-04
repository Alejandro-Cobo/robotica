from pyrobot.brain import Brain

import math

class BrainTestNavigator(Brain):
  last_error = 0
  avoid = False
  search_line = True

  def setup(self):
    pass

  def step(self):
    hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
    print "I got from the simulation",hasLine,lineDistance,searchRange,self.last_error

    front = min([s.distance() for s in self.robot.range["front"]])   

    # Search for a line
    if hasLine:
       self.search_line = False
    if self.search_line:
       if front < 0.5 or self.robot.range[5].distance() < 0.5 or self.robot.range[6].distance() < 0.5:
          self.move(1,0)
       else:
          self.move(0.5,0)
       return      

    # Avoid obstacles
    if front < 0.5:
       self.avoid = True
       self.move(0, 0.5)
    if self.avoid:
       if self.robot.range[5].distance() < 0.5:
          self.move(0,0.5)
       elif self.robot.range[6].distance() < 0.5:
          self.move(0.3,0.3)
       elif self.robot.range[7].distance() < 0.5:
          if hasLine:
             self.avoid = False
          self.move(0.3,-0.5)
       elif self.robot.range[7].distance() > 0.5 and self.robot.range[7].distance() < 0.8:
          self.move(0.3,-0.3)

    # Follow the line:
    if ( hasLine and not(self.avoid) ):
      kd = (lineDistance/searchRange) / 8
      d = (lineDistance - self.last_error) / searchRange
      self.last_error = lineDistance
      tv = (lineDistance/searchRange) + kd*d
      fv = max( 0, 1-abs(tv*1.5) - abs(d) )
      self.move(fv, tv)
    elif ( not(hasLine) and not(self.avoid) ):
      self.move(0, self.last_error / searchRange)
 
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
