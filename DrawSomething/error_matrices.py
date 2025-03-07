
import math
from DrawSomething.geometry_utils import line_segment_distance
# error matrices
def error_line(points):
    if len(points) < 2:
        return 9999
    x1, y1 = points[0]
    x2, y2 = points[-1]
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-5:
        return 9999
    total_dist = 0
    for (x, y) in points:
        dist = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length
        total_dist += dist
    return total_dist / len(points)


def error_circle(points):
    if len(points) < 3:
        return 9999
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    dists = [math.hypot(x - cx, y - cy) for (x, y) in points]
    if not dists:
        return 9999
    r_avg = sum(dists) / len(dists)
    total_error = sum(abs(d - r_avg) for d in dists)
    return total_error / len(points)


def error_ellipse(points):
    if len(points) < 3:
        return 9999
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width < 1 or height < 1:
        return 9999
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    rx = width / 2
    ry = height / 2
    total_error = 0
    for (x, y) in points:
        dx = x - cx
        dy = y - cy
        val = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
        total_error += abs(val - 1.0)
    return 25 * (total_error / len(points))


def error_rectangle(points):
    if len(points) < 2:
        return 9999
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if (max_x - min_x) < 1 or (max_y - min_y) < 1:
        return 9999
    edges = [
        (min_x, min_y, max_x, min_y),
        (max_x, min_y, max_x, max_y),
        (max_x, max_y, min_x, max_y),
        (min_x, max_y, min_x, min_y)
    ]
    total_dist = 0
    for (px, py) in points:
        dists = []
        for (x1, y1, x2, y2) in edges:
            d = line_segment_distance(px, py, x1, y1, x2, y2)
            dists.append(d)
        min_dist_for_point = min(dists)
        total_dist += min_dist_for_point
    avg_dist = total_dist / len(points)
    return 1.2 * avg_dist + 1


def error_triangle(points):
    if len(points) < 3:
        return 9999
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width < 1 or height < 1:
        return 9999
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    radius = min(width, height) / 2
    v1 = (cx, cy - radius)
    v2 = (cx - (radius * math.sqrt(3) / 2), cy + radius / 2)
    v3 = (cx + (radius * math.sqrt(3) / 2), cy + radius / 2)

    def line_dist(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        denom = dx * dx + dy * dy
        if denom == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / denom
        if t < 0:
            return math.hypot(px - x1, py - y1)
        elif t > 1:
            return math.hypot(px - x2, py - y2)
        projx = x1 + t * dx
        projy = y1 + t * dy
        return math.hypot(px - projx, py - projy)

    total_dist = 0
    for (px, py) in points:
        d1 = line_dist(px, py, v1[0], v1[1], v2[0], v2[1])
        d2 = line_dist(px, py, v2[0], v2[1], v3[0], v3[1])
        d3 = line_dist(px, py, v3[0], v3[1], v1[0], v1[1])
        dist_pt = min(d1, d2, d3)
        total_dist += dist_pt
    return (0.5 * (total_dist / len(points)) + 1)


def count_close_point_triangle(points, epsilon_ratio=0.01):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    epsilon = epsilon_ratio * (min(width, height))

    count_points_on_edge1 = 0
    count_points_on_edge2 = 0
    count_points_on_edge3 = 0
    count_points_on_edge4 = 0

    for point in points:
        if abs(point[1] - max_y) < epsilon:
            count_points_on_edge1 += 1
        if abs(point[0] - max_x) < epsilon:
            count_points_on_edge2 += 1
        if abs(point[1] - min_y) < epsilon:
            count_points_on_edge3 += 1
        if abs(point[0] - min_x) < epsilon:
            count_points_on_edge4 += 1

    counter_arr = [count_points_on_edge1, count_points_on_edge2, count_points_on_edge3, count_points_on_edge4]
    selected_count = max(counter_arr)
    selected_edge = counter_arr.index(selected_count)

    if selected_count >= 3 and selected_edge == 0:
        return selected_count, "Upper"
    elif selected_count >= 3 and selected_edge == 1:
        return selected_count, "Right"
    elif selected_count >= 3 and selected_edge == 2:
        return selected_count, "Lower"
    elif selected_count >= 3 and selected_edge == 3:
        return selected_count, "Left"
    else:
        return selected_count, "None"