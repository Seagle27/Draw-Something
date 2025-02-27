import tkinter as tk
import math

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shape Smoothing & Best-Fit Correction")

        # Current drawing settings
        self.current_color = "black"
        self.eraser_mode = False

        # Store all completed strokes here
        # Each element: {"points": [...], "color": str, "eraser": bool}
        self.drawn_strokes = []

        # Temporary list for the current stroke
        self.current_points = []

        # List of color buttons (circular) to display on canvas
        self.color_circles = []
        self.is_selecting_color = False

        # Create main canvas
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # We'll just define 5 color buttons at the top
        self.draw_color_buttons()
        self.draw_eraser_button()

        # Tools menu for eraser
        # self.menu_bar = tk.Menu(self.root)
        # tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        # tools_menu.add_command(label="Eraser", command=self.activate_eraser)
        # self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        # self.root.config(menu=self.menu_bar)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Bind Ctrl+S for smoothing & correction
        self.root.bind("<Control-s>", self.on_ctrl_s)

    # ---------------------
    # Color Buttons
    # ---------------------

    def draw_color_buttons(self):
        color_options = ["black", "red", "green", "blue", "yellow"]
        start_x = 40
        y = 40
        radius = 15
        spacing = 60

        for i, color in enumerate(color_options):
            cx = start_x + i * spacing
            cy = y
            self.canvas.create_oval(cx - radius, cy - radius,
                                    cx + radius, cy + radius,
                                    outline=color, fill=color)

            self.color_circles.append({
                "color": color,
                "center": (cx, cy),
                "radius": radius
            })

    def draw_eraser_button(self):
        # Define the eraser button's properties (position, radius, color, etc.)
        cx, cy = 340, 40  # for example, place it below the color buttons
        radius = 15
        # Draw a circle with a border to represent the eraser button
        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                outline="gray", fill="lightgray")
        # Optionally, add a label (like "E") to indicate it's an eraser
        self.canvas.create_text(cx, cy, text="E", fill="black")
        # Save the button's properties to check for clicks later
        self.eraser_button = {"center": (cx, cy), "radius": radius}

    def check_color_button_click(self, x, y):
        for circle in self.color_circles:
            cx, cy = circle["center"]
            r = circle["radius"]
            dist = math.hypot(x - cx, y - cy)
            if dist <= r:
                self.set_color(circle["color"])
                return True
        return False

    def check_eraser_button_click(self, x, y):
        cx, cy = self.eraser_button["center"]
        r = self.eraser_button["radius"]
        if math.hypot(x - cx, y - cy) <= r:
            self.activate_eraser()
            return True
        return False

    def set_color(self, color):
        self.current_color = color
        self.eraser_mode = False

    def activate_eraser(self):
        self.eraser_mode = True

    # ---------------------
    # Mouse Event Handlers
    # ---------------------

    # def on_button_press(self, event):
    #     if self.check_color_button_click(event.x, event.y):
    #         self.is_selecting_color = True
    #     else:
    #         self.is_selecting_color = False
    #         self.current_points = [(event.x, event.y)]

    def on_button_press(self, event):
        if self.check_color_button_click(event.x, event.y):
            self.is_selecting_color = True
        elif self.check_eraser_button_click(event.x, event.y):
            # We have activated eraser mode; no new stroke needed.
            return
        else:
            self.is_selecting_color = False
            self.current_points = [(event.x, event.y)]

    def on_mouse_move(self, event):
        if self.is_selecting_color:
            return

        if not self.current_points:
            return

        x, y = event.x, event.y
        self.current_points.append((x, y))

        if len(self.current_points) > 1:
            x1, y1 = self.current_points[-2]
            x2, y2 = self.current_points[-1]
            if self.eraser_mode:
                self.canvas.create_line(x1, y1, x2, y2, fill="white", width=10)
            else:
                self.canvas.create_line(x1, y1, x2, y2, fill=self.current_color, width=2)

    def on_button_release(self, event):
        if self.is_selecting_color:
            return

        if self.current_points:
            stroke_data = {
                "points": self.current_points[:],
                "color": self.current_color,
                "eraser": self.eraser_mode
            }
            self.drawn_strokes.append(stroke_data)
        self.current_points = []

    def on_ctrl_s(self, event):
        """When Ctrl+S is pressed, clear and re-draw everything with smoothing & best-fit shape."""
        self.canvas.delete("all")

        for stroke in self.drawn_strokes:
            points = stroke["points"]
            color = stroke["color"]
            eraser_mode = stroke["eraser"]

            if eraser_mode:
                # Redraw the eraser stroke as a white line
                self.draw_freehand(points, "white", width=10)
            else:
                # 1) Smooth the points (Chaikin for nicer curves)
                smoothed = self.chaikin_smoothing(points, iterations=2)

                # 2) Best-Fit detection
                shape = self.best_fit_shape(smoothed)

                # 3) Draw result
                if shape == "line":
                    self.draw_line(smoothed, color)
                elif shape == "circle":
                    self.draw_circle(smoothed, color)
                elif shape == "ellipse":
                    self.draw_ellipse(smoothed, color)
                elif shape == "rectangle":
                    self.draw_rectangle(smoothed, color)
                elif shape == "triangle":
                    self.draw_triangle(smoothed, color)
                else:
                    # Default = abstract (or unrecognized)
                    self.draw_freehand(smoothed, color)

        # Re-draw color buttons
        self.draw_color_buttons()
        self.draw_eraser_button()

    # ---------------------
    # Smoothing (Chaikin)
    # ---------------------

    def chaikin_smoothing(self, points, iterations=2):
        if len(points) < 3:
            return points

        new_points = points
        for _ in range(iterations):
            new_points = self.chaikin_once(new_points)
        return new_points

    def chaikin_once(self, points):
        if len(points) < 3:
            return points

        new_pts = [points[0]]  # keep first point
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            qx = 0.75 * p1[0] + 0.25 * p2[0]
            qy = 0.75 * p1[1] + 0.25 * p2[1]
            rx = 0.25 * p1[0] + 0.75 * p2[0]
            ry = 0.25 * p1[1] + 0.75 * p2[1]

            new_pts.append((qx, qy))
            new_pts.append((rx, ry))
        new_pts.append(points[-1])  # keep last point
        return new_pts

    # ---------------------
    # Best-Fit Shape
    # ---------------------
    def line_segment_distance(self, px, py, x1, y1, x2, y2):
        """
        Returns the minimum distance from point (px, py) to the line segment from (x1, y1) to (x2, y2).
        """
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy

        # If the segment length is zero
        if seg_len_sq == 0:
            # The distance is simply from the point to (x1, y1)
            return math.hypot(px - x1, py - y1)

        # Compute t, the normalized projection of (px, py) onto the segment
        # t = 0 => perpendicular to (x1, y1), t = 1 => perpendicular to (x2, y2), between 0 and 1 => lies on segment
        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq

        if t < 0:
            # Closest to (x1, y1)
            return math.hypot(px - x1, py - y1)
        elif t > 1:
            # Closest to (x2, y2)
            return math.hypot(px - x2, py - y2)
        else:
            # Projection falls within the segment
            projx = x1 + t * dx
            projy = y1 + t * dy
            return math.hypot(px - projx, py - projy)

    def best_fit_shape(self, points):
        """
        Attempts to match several shapes (Line, Circle, Ellipse, Rectangle, Triangle)
        and returns the shape with the lowest average error (if it is below a certain threshold).
        Otherwise, returns "abstract".
        """
        candidates = {
            "line": self.error_line(points),
            "circle": self.error_circle(points),
            "ellipse": self.error_ellipse(points),
            "rectangle": self.error_rectangle(points),
            "triangle": self.error_triangle(points)
        }

        # Define a threshold such that if the average error is too high, the shape is not selected
        THRESHOLD = 7.0
        #THRESHOLD = 9

        best_shape = "abstract"
        best_error = float("inf")

        for shape_name, err in candidates.items():
            print(shape_name, err)
            if err < best_error:
                best_error = err
                best_shape = shape_name
        print("best_shape:", best_shape)

        if best_error > THRESHOLD:
            return "abstract"
        return best_shape

    # -- Error metrics for shapes --

    def error_line(self, points):
        """
        Calculates the average perpendicular distance (or RMS) from each point to the line between the first and last point.
        """
        if len(points) < 2:
            return 9999

        x1, y1 = points[0]
        x2, y2 = points[-1]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-5:
            return 9999  # almost no line

        # Sum the distances
        total_dist = 0
        for (x, y) in points:
            dist = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length
            total_dist += dist

        return total_dist / len(points)

    def error_circle(self, points):
        """
        Computes a circle (center + radius) based on the bounding box or average of the points,
        then checks the average deviation of each point's distance from the center relative to the average radius.
        """
        if len(points) < 3:
            return 9999

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Center of the circle
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        # Radius = average distance from the center
        dists = [math.hypot(x - cx, y - cy) for (x, y) in points]
        if not dists:
            return 9999
        r_avg = sum(dists) / len(dists)

        # Compute error = average |distance_i - r_avg|
        total_error = sum(abs(d - r_avg) for d in dists)
        return total_error / len(points)

    def error_ellipse(self, points):
        """
        Fit an "ellipse" based on the bounding box:
          center = (cx, cy), rx = (max_x - min_x) / 2, ry = (max_y - min_y) / 2.
        Then measure how close each (x, y) satisfies (dx^2)/(rx^2) + (dy^2)/(ry^2) ≈ 1.
        The error is chosen as the average |value - 1|.
        """
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
            val = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)  # ideally ~1
            total_error += abs(val - 1.0)

        return 25 * (total_error / len(points))

    def error_rectangle(self, points):
        """
        Improves the error calculation for a rectangle:
        Defines an axis-aligned rectangle based on the bounding box,
        then computes for each point the minimum distance to one of the four sides.
        The error is the average of these distances.
        """
        if len(points) < 2:
            return 9999  # large error if no real data

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # If the bounding box is tiny, return a high error to avoid recognizing it as a rectangle
        if (max_x - min_x) < 1 or (max_y - min_y) < 1:
            return 9999

        # Define the 4 edges of the rectangle
        edges = [
            (min_x, min_y, max_x, min_y),  # top
            (max_x, min_y, max_x, max_y),  # right
            (max_x, max_y, min_x, max_y),  # bottom
            (min_x, max_y, min_x, min_y)   # left
        ]

        total_dist = 0
        for (px, py) in points:
            # Minimum distance for this point to any edge
            dists = []
            for (x1, y1, x2, y2) in edges:
                d = self.line_segment_distance(px, py, x1, y1, x2, y2)
                dists.append(d)
            min_dist_for_point = min(dists)
            total_dist += min_dist_for_point

        avg_dist = total_dist / len(points)
        return 1.2*avg_dist

    def error_triangle(self, points):
        """
        Assume an equilateral triangle inscribed in the bounding box,
        then measure the distance from each point to the triangle's perimeter.
        """
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

        # Equilateral triangle vertices:
        v1 = (cx, cy - radius)
        v2 = (cx - (radius * math.sqrt(3) / 2), cy + radius / 2)
        v3 = (cx + (radius * math.sqrt(3) / 2), cy + radius / 2)

        # Helper: distance from point to line (x1, y1) -> (x2, y2)
        def line_dist(px, py, x1, y1, x2, y2):
            dx = x2 - x1
            dy = y2 - y1
            denom = dx * dx + dy * dy
            if denom == 0:
                return math.hypot(px - x1, py - y1)

            # Compute the perpendicular projection
            t = ((px - x1) * dx + (py - y1) * dy) / denom
            if t < 0:
                # Closer to the first vertex
                return math.hypot(px - x1, py - y1)
            elif t > 1:
                # Closer to the second vertex
                return math.hypot(px - x2, py - y2)
            else:
                projx = x1 + t * dx
                projy = y1 + t * dy
                return math.hypot(px - projx, py - projy)

        # Compute average distance from the three edges
        total_dist = 0
        for (px, py) in points:
            d1 = line_dist(px, py, v1[0], v1[1], v2[0], v2[1])
            d2 = line_dist(px, py, v2[0], v2[1], v3[0], v3[1])
            d3 = line_dist(px, py, v3[0], v3[1], v1[0], v1[1])
            # Minimal distance from one of the edges
            dist_pt = min(d1, d2, d3)
            total_dist += dist_pt

        return 0.3*(total_dist / len(points))

    # -------------------
    # Drawing Functions
    # -------------------

    def draw_freehand(self, points, color, width=2):
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def draw_line(self, points, color):
        x1, y1 = points[0]
        x2, y2 = points[-1]
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

    def draw_circle(self, points, color):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y
        diameter = max(width, height)

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        left = cx - diameter / 2
        right = cx + diameter / 2
        top = cy - diameter / 2
        bottom = cy + diameter / 2

        self.canvas.create_oval(left, top, right, bottom, outline=color, fill="", width=2)

    def draw_ellipse(self, points, color):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self.canvas.create_oval(min_x, min_y, max_x, max_y, outline=color, fill="", width=2)

    def draw_rectangle(self, points, color):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        self.canvas.create_rectangle(min_x, min_y, max_x, max_y, outline=color, fill="", width=2)

    def draw_triangle(self, points, color):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        radius = min(width, height) / 2.0

        v1 = (cx, cy - radius)
        v2 = (cx - (radius * math.sqrt(3) / 2), cy + radius / 2)
        v3 = (cx + (radius * math.sqrt(3) / 2), cy + radius / 2)

        self.canvas.create_line(v1[0], v1[1], v2[0], v2[1], fill=color, width=2)
        self.canvas.create_line(v2[0], v2[1], v3[0], v3[1], fill=color, width=2)
        self.canvas.create_line(v3[0], v3[1], v1[0], v1[1], fill=color, width=2)

def main():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
