import math

def calc_triangle_angle(A, B, C):
  def length_square(p1, p2):
    x_diff = p1[0]-p2[0]
    y_diff = p1[1]-p2[1]
    return x_diff ** 2 + y_diff ** 2

  # Square of lengths
  a2 = length_square(B, C)
  b2 = length_square(A, C)
  c2 = length_square(A, B)

  a = math.sqrt(a2)
  b = math.sqrt(b2)
  c = math.sqrt(c2)

  alpha = math.acos((b2 + c2 - a2)/(2*b*c))
  beta = math.acos((a2 + c2 - b2)/(2*a*c))
  gamma = math.acos((a2 + b2 - c2)/(2*a*b))

  alpha = alpha * 180 / math.pi
  beta = beta * 180 / math.pi
  gamma = gamma * 180 / math.pi

  print(alpha)
  print(beta)
  print(gamma)


# Full size pool table dimensions in inches
table_width = 50.25
table_length = 100.4375
ball_width = 2.125

pocket_ball_1 = .25
pocket_ball_2 = .625

# 20" pool table
#table_width = 9.375
#table_length = 17.25
#ball_width = 1.0
#pocket_ball_1 = .25
#pocket_ball_2 = .675



# Starting top left ball
A = (ball_width, ball_width)

# Opposite diagonal pocket
B = (table_length-pocket_ball_1, table_width-pocket_ball_2)
C = (table_length-pocket_ball_2, table_width-pocket_ball_1)

print(math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2))
# calc_triangle_angle(A, B, C)
