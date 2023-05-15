# find points in a specific distance
def find_points(x, y, m, distance):
    point_b = (x + dx(distance, m), y + dy(distance, m))
    other_possible_point_b = (x - dx(distance, m), y - dy(distance, m))  # going the other way
    return point_b, other_possible_point_b


def dy(distance, m):
    return m * dx(distance, m)


def dx(distance, m):
    return sqrt(distance ** 2 / (m ** 2 + 1))
