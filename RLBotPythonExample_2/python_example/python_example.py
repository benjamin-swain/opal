import math
import pygame
import neurolab as nl

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket


class PythonExample(BaseAgent):

	def initialize_agent(self):
        #This runs once before the bot starts up
		self.controller_state = SimpleControllerState()
		
		# load nn
		self.nn = nl.load('rl_nn.net')
		
		#persistent count
		self.count = 1
		
		#PS4 Controller
		pygame.init()
		pygame.joystick.init()

	def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
	
		#PS4 Controller
		ps4 = pygame.joystick.Joystick(0)
		ps4.init()
		events = pygame.event.get()
		# for event in events:
			# if event.type == pygame.JOYBUTTONDOWN:
				# print("Button Pressed")
				# if ps4.get_button(6):
					# print('L2')
				# elif ps4.get_button(7):
					# print('R2')
			# elif event.type == pygame.JOYBUTTONUP:
				# print("Button Released")
		
		# --PS4 Controls--
		# a4, R2, throttle (-1 nopress, 1 pressed)
		# a5, L2, reverse (-1 nopress, 1 pressed)
		# a0, LS, steer (-1 left, 1 right)
		# a1, LS, pitch (-1 up, 1 down)
		# b3, triangle, roll left
		# b2, circle, roll right
		# b1, X, jump
		# b5, R1, boost
		# b4, L1, handbrake
		
		# --Throttle Calculation--
		# 1 full throttle, t=1, r=-1, T=(t-r)/2 = (1+1)/2 = 1
		# 0 neither, t=-1, r=-1, T=(t-r)/2 = (-1+1)/2 = 0
		# -1 full reverse, t=-1, r=1, T=(t-r)/2 = (-1-1)/2 = -1
		# 0.5 half throttle, t=0, r=-1, T=(t-r)/2 = (0+1)/2 = 0.5
		
		# --Roll Calculation--
		# 1 roll right, L=0, R=1, T=R-L = 1-0 = 1 
		# -1 roll left, L=1, R=0, T= 0-1 = -1
		# 0 none, L=0, R=0, T= 0-0 = 0
		# 0 both, L=1, R=1, T= 1-1 = 0
				

		human = packet.game_cars[1]
		ball = packet.game_ball
		# inputs 
		h_x = human.physics.location.x
		h_y = human.physics.location.y
		h_z = human.physics.location.z
		h_pitch = human.physics.rotation.pitch
		h_yaw = human.physics.rotation.yaw
		h_roll = human.physics.rotation.roll
		h_vx = human.physics.velocity.x
		h_vy = human.physics.velocity.y
		h_vz = human.physics.velocity.z
		h_avx = human.physics.angular_velocity.x
		h_avy = human.physics.angular_velocity.y
		h_avz = human.physics.angular_velocity.z
		h_has_wheel_contact = human.has_wheel_contact
		h_super_sonic = human.is_super_sonic
		h_jumped = human.jumped
		h_double_jumped = human.double_jumped
		h_boost = human.boost
		b_x = ball.physics.location.x
		b_y = ball.physics.location.y
		b_z = ball.physics.location.z
		b_vx = ball.physics.velocity.x
		b_vy = ball.physics.velocity.y
		b_vz = ball.physics.velocity.z
		b_avx = ball.physics.angular_velocity.x
		b_avy = ball.physics.angular_velocity.y
		b_avz = ball.physics.angular_velocity.z
		# outputs
		h_throttle = (ps4.get_axis(4) - ps4.get_axis(5))/2.0
		h_steer = ps4.get_axis(0)
		h_pitch = ps4.get_axis(1)
		# yaw is the same as steer from PS4 controller's point of view
		h_roll = ps4.get_button(2) - ps4.get_button(3)
		h_jump = ps4.get_button(1)
		h_boost = ps4.get_button(5)
		h_brake = ps4.get_button(4)
		
		input_list = [h_x, h_y, h_z, h_pitch, h_yaw, h_roll, h_vx, h_vy, h_vz, h_avx, h_avy, h_avz, h_has_wheel_contact, h_super_sonic, h_jumped, h_double_jumped, h_boost, b_x, b_y, b_z, b_vx, b_vy, b_vz, b_avx, b_avy, b_avz]
				
		# print(packet.game_info.is_round_active)
		# if packet.game_info.is_round_active:
			# input_output_list = [h_x, h_y, h_z, h_pitch, h_yaw, h_roll, h_vx, h_vy, h_vz, h_avx, h_avy, h_avz, h_has_wheel_contact, h_super_sonic, h_jumped, h_double_jumped, h_boost, b_x, b_y, b_z, b_vx, b_vy, b_vz, b_avx, b_avy, b_avz, h_throttle, h_steer, h_pitch, h_roll, h_jump, h_boost, h_brake]
			# shorter = ['{:.3f}'.format(x) for x in input_output_list]
			# shorterer = " ".join(shorter)
			# print(shorterer)
			# Record data for training
			# nn_data = open("nn_data.txt","a")
			# nn_data.write(shorterer + '\n')
			# nn_data.close() #to change file access modes
	
	
		# EXAMPLE CONTROLS
		ball_location = Vector2(packet.game_ball.physics.location.x, packet.game_ball.physics.location.y)

		my_car = packet.game_cars[self.index]
		car_location = Vector2(my_car.physics.location.x, my_car.physics.location.y)
		car_direction = get_car_facing_vector(my_car)
		car_to_ball = ball_location - car_location

		steer_correction_radians = car_direction.correction_to(car_to_ball)

		if steer_correction_radians > 0:
			# Positive radians in the unit circle is a turn to the left.
			turn = -1.0  # Negative value for a turn to the left.
			action_display = "turn left"
		else:
			turn = 1.0
			action_display = "turn right"

			
		# DOUBLE JUMP
		# self.count += 1
		# print(self.count)
		# if self.count <= 100:
			# self.controller_state.jump = 0
		# if self.count > 170 and self.count <= 180:
			# self.controller_state.jump = 1
		# if self.count > 180 and self.count <= 200:
			# self.controller_state.jump = 0
		# if self.count > 200 and self.count <= 220:
			# self.controller_state.jump = 1
		# if self.count > 220:
			# self.controller_state.jump = 0
			# self.count = 1
			
		
			
		self.controller_state.throttle = 1.0
		self.controller_state.steer = turn
		#self.controller_state.steer = 0.6
		#self.controller_state.yaw = 0.49
		
		# NEURAL NET CONTROLS
		# print(input_list)
		controls = self.nn.sim([input_list])
		print(controls)

		draw_debug(self.renderer, my_car, packet.game_ball, action_display)

		return self.controller_state

class Vector2:
	def __init__(self, x=0, y=0):
		self.x = float(x)
		self.y = float(y)

	def __add__(self, val):
		return Vector2(self.x + val.x, self.y + val.y)

	def __sub__(self, val):
		return Vector2(self.x - val.x, self.y - val.y)

	def correction_to(self, ideal):
		# The in-game axes are left handed, so use -x
		current_in_radians = math.atan2(self.y, -self.x)
		ideal_in_radians = math.atan2(ideal.y, -ideal.x)

		correction = ideal_in_radians - current_in_radians

		# Make sure we go the 'short way'
		if abs(correction) > math.pi:
			if correction < 0:
				correction += 2 * math.pi
			else:
				correction -= 2 * math.pi

		return correction


def get_car_facing_vector(car):
	pitch = float(car.physics.rotation.pitch)
	yaw = float(car.physics.rotation.yaw)

	facing_x = math.cos(pitch) * math.cos(yaw)
	facing_y = math.cos(pitch) * math.sin(yaw)

	return Vector2(facing_x, facing_y)

def draw_debug(renderer, car, ball, action_display):
	renderer.begin_rendering()
	# draw a line from the car to the ball
	renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
	# print the action that the bot is taking
	renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
	renderer.end_rendering()
