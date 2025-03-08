"""
DrawingApp Class - with Camera/Video
"""

import tkinter as tk
import math
import cv2
import numpy as np
from DrawSomething.segmentation import HandSegmentation
from DrawSomething import gesture_recognition as gest

# Global Constants & Config
from DrawSomething import constants

# Utility / Standalone Functions
from DrawSomething.overlay_utils import apply_brush_mask
from DrawSomething.print_on_frame import print_gesture_on_frame,print_color_on_frame
from DrawSomething.fingertip_detection import find_fingertip,smooth_fingertip,is_valid_fingertip,detect_fingertip,preprocess_mask
from DrawSomething.geometry_utils import rotate_point, chaikin_smoothing
from DrawSomething.shape_detection import best_fit_shape


# =====================================
# DrawingApp Class Definition
# =====================================

class DrawingApp:
    # ---------------------
    # init functions
    # ---------------------
    def __init__(self, root):
        # ---------
        # General
        # ---------
        self.root = root
        self.root.title("Shape Smoothing & Best-Fit Correction")
        # self.screen_width = root.winfo_screenwidth()
        # self.screen_height = root.winfo_screenheight()
        #self.root.geometry(f"{self.screen_width // 2}x{self.screen_height}+0+0")
        self.root.geometry(f"{constants.FRAME_WIDTH}x{constants.FRAME_HEIGHT}+0+0")
        # ---------
        # Video
        # ---------
        self.cap = cv2.VideoCapture(0)
        self.mask_func = HandSegmentation(self.cap)

        self.svm_model = gest.SvmModel(constants.SVM_MODEL_PATH)
        self.stabilizer = gest.GestureStabilizer(constants.WIN_SIZE, constants.MIN_CHANGE_FRAME)

        self.fingertip_down = False
        self.root.after(1, self.update_frame)  # Start an update loop every 10ms
        # ---------
        # Gestures
        # ---------
        self.current_gesture = "index_finger"  # can be: index_finger, up_thumb, open_hand, close_hand, three_fingers
        self.prev_gesture = "close_hand"
        # ------------
        # Drawing Data
        # ------------
        # Current state
        self.current_color = {"black": (0, 0, 0)}
        self.current_width = 2
        self.eraser_mode = False
        self.eraser_type = None  # can be "Normal" or "Stroke"
        self.stability_counter = 0
        self.curr_fingertip = None

        # Drawing data
        self.drawn_strokes = []
        self.current_points = []
        #self.num_strokes = 0

        # Previous state
        self.prev_color = None
        self.prev_width = None
        self.prev_eraser_mode = None
        self.last_eraser = None
        self.fingertip_history = []
        self.prev_fingertip = None

        # ------------
        # Buttons Data
        # ------------
        # Buttons Location
        self.color_circles = []
        self.eraser_button = {}
        self.eraser_stroke_button = {}
        self.clear_button = {}
        self.width_circles = []
        self.set_positions_dict()

        # Create main canvas
        self.canvas = tk.Canvas(self.root, bg="white", width=constants.FRAME_WIDTH, height=constants.FRAME_HEIGHT)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.static_overlay = None
        self.mask = None
        self.create_buttons_overlay()
        self.eraser_partial_mask = np.zeros((constants.FRAME_HEIGHT, constants.FRAME_WIDTH), dtype=np.uint8)

    # =============================
    # Buttons / UI Helpers
    # =============================

    def set_positions_dict(self):
        """
        Pre-calculates and stores the positions (center, radius) of color buttons, eraser button,
        stroke-eraser button, clear button, and width buttons on the UI overlay.
        """
        x = constants.START_X_POS
        colors = constants.COLORS_OPTIONS

        # Color Buttons
        for color_key, color_value in colors.items():
            self.color_circles.append({
                "color": color_key,
                "center": (x, constants.Y_POS),
                "radius": constants.BUTTON_RADIUS
            })
            x += constants.BUTTON_SPACING

        # Clear Button
        x = constants.X_CLEAR
        y = constants.Y_CLEAR
        self.clear_button = {
            "center": (x, y),
            "radius": constants.BUTTON_RADIUS
        }

        # Width Circles
        y = constants.START_Y_POS
        #x = constants.X_POS
        x = constants.START_X_POS
        for line_width in constants.WIDTH_OPTIONS:
            self.width_circles.append({
                "width": line_width,
                "center": (x, y),
                "radius": constants.BUTTON_RADIUS
            })
            #y += constants.BUTTON_SPACING
            x += constants.BUTTON_SPACING

        # Eraser Button
        self.eraser_button = {
            "center": (x, y),
            "radius": constants.BUTTON_RADIUS
        }
        x += constants.BUTTON_SPACING

        # Stroke-Eraser Button
        self.eraser_stroke_button = {
            "center": (x, y),
            "radius": constants.BUTTON_RADIUS
        }
        x += constants.BUTTON_SPACING

    def create_buttons_overlay(self):
        """
        Creates a static overlay image with drawn color buttons, eraser buttons,
        and width buttons. Also creates a mask that helps in rendering this overlay
        onto each video frame.
        """
        overlay = np.zeros((constants.FRAME_HEIGHT, constants.FRAME_WIDTH, 3), dtype=np.uint8)
        mask = np.zeros((constants.FRAME_HEIGHT, constants.FRAME_WIDTH), dtype=np.uint8)
        gray = (128, 128, 128)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text_color = (255, 255, 255)
        font_scale = 0.5

        # --- Draw Color Buttons (with eraser and clear) ---
        x = constants.START_X_POS
        for color_key, color_value in constants.COLORS_OPTIONS.items():
            radius = constants.BUTTON_RADIUS
            x1 = max(x - 22, 0)
            x2 = min(x + 22 + 1, constants.FRAME_WIDTH)
            y1 = max(constants.Y_POS - 22, 0)
            y2 = min(constants.Y_POS + 22 + 1, constants.FRAME_HEIGHT)


            if list(self.current_color.keys())[0] == color_key and not self.eraser_mode:
                radius = constants.BUTTON_ROAIUS_OF_SELECTED

            cv2.circle(overlay, (x, constants.Y_POS), radius, color_value, -1)
            cv2.circle(mask, (x, constants.Y_POS), radius, 255, -1)

            if color_key == "black":
                overlay, mask_brush = apply_brush_mask(
                    constants.BRUSH_BLACK_ICON, overlay, x, constants.Y_POS, type="white", threshold=128
                )
            else:
                overlay,_ = apply_brush_mask(
                    constants.BRUSH_BLACK_ICON, overlay, x, constants.Y_POS, type="black", threshold=128
                )
            region = mask[y1:y2, x1:x2]
            region[mask_brush == 0] = 255
            x += constants.BUTTON_SPACING

        # Clear Button
        x = constants.X_CLEAR
        y = constants.Y_CLEAR
        cv2.circle(overlay, (x, y), constants.BUTTON_RADIUS, gray, -1)
        cv2.circle(mask, (x, y), constants.BUTTON_RADIUS, 255, -1)

        text = "clear"
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x - text_size[0] // 2
        cv2.putText(overlay, text, (text_x, y + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # --- Draw Width Buttons ---
        y = constants.START_Y_POS
        x = constants.START_X_POS
        for line_width in constants.WIDTH_OPTIONS:
            radius = constants.BUTTON_RADIUS
            if self.current_width == line_width:
                radius = constants.BUTTON_ROAIUS_OF_SELECTED
            circle_color = (200, 200, 200)
            cv2.circle(overlay, (x, y), radius, circle_color, -1)
            cv2.circle(mask, (x, y), radius, 255, -1)
            margin = 10
            start_point = (x - constants.BUTTON_RADIUS + margin, y)
            end_point = (x + constants.BUTTON_RADIUS - margin, y)
            line_color = (0, 0, 0)
            cv2.line(overlay, start_point, end_point, line_color, line_width)
            x += constants.BUTTON_SPACING

        radius = constants.BUTTON_RADIUS
        if self.eraser_mode and self.eraser_type == "Normal":
            radius = constants.BUTTON_ROAIUS_OF_SELECTED
        cv2.circle(overlay, (x, y), radius, gray, -1)
        cv2.circle(mask, (x, y), radius, 255, -1)
        overlay = apply_brush_mask(constants.ERASER_BLACK_ICON, overlay, x, y, type="black", invert=True,
                                   threshold=128)[0]
        overlay = apply_brush_mask(constants.ERASER_WHITE_ICON, overlay, x, y, type="white", invert=False,
                                   threshold=128)[0]
        x += constants.BUTTON_SPACING

        # Stroke-Eraser Button
        radius = constants.BUTTON_RADIUS
        if self.eraser_mode and self.eraser_type == "Stroke":
            radius = constants.BUTTON_ROAIUS_OF_SELECTED
        cv2.circle(overlay, (x, y), radius, gray, -1)
        cv2.circle(mask, (x, y), radius, 255, -1)
        overlay = \
        apply_brush_mask(constants.ERASER_S_WHITE_ICON, overlay, x, y, type="white", invert=False,
                         threshold=128)[0]
        overlay = \
        apply_brush_mask(constants.ERASER_S_BLACK_ICON, overlay, x, y, type="black", invert=True,
                         threshold=128)[0]

        self.mask = mask
        self.static_overlay = overlay

    # =============================
    # Button-Click Checks
    # =============================

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
            self.activate_eraser("Normal")
            return True
        return False

    def check_S_eraser_button_click(self, x, y):
        cx, cy = self.eraser_stroke_button["center"]
        r = self.eraser_stroke_button["radius"]
        if math.hypot(x - cx, y - cy) <= r:
            self.activate_eraser("Stroke")
            return True
        return False

    def check_clear_button_click(self, x, y):
        cx, cy = self.clear_button["center"]
        r = self.clear_button["radius"]
        if math.hypot(x - cx, y - cy) < r:
            self.clear_all_drawings()
            return True
        return False

    def check_width_button_click(self, x, y):
        for circle in self.width_circles:
            # if x > constants.BUTTON_RADIUS and x < threshold:
            #     x = x - threshold
            cx, cy = circle["center"]
            r = circle["radius"]
            dist = math.hypot(x - cx, y - cy)
            if dist <= r:
                self.set_width(circle["width"])
                return True
        return False

    # ==============================
    # Set / Update Methods
    # ==============================

    def set_color(self, color_key):
        self.prev_color = self.current_color
        color_rgb_value = constants.COLORS_OPTIONS[color_key]
        self.current_color = {color_key: color_rgb_value}
        self.eraser_mode = False

    def set_width(self, width):
        self.prev_width = self.current_width
        self.current_width = width

    def clear_all_drawings(self):
        if len(self.drawn_strokes)!=0:
            print("Cleaning the canvas...")
            self.canvas.delete("all")
            self.drawn_strokes.clear()
            self.eraser_partial_mask = np.zeros((constants.FRAME_HEIGHT, constants.FRAME_WIDTH), dtype=np.uint8)

    def activate_eraser(self,type):
        self.prev_eraser_mode = self.eraser_mode
        self.last_eraser = self.eraser_type
        self.eraser_type = type
        self.eraser_mode = True

    def update_gesture(self, gesture):
        self.prev_gesture = self.current_gesture
        self.current_gesture = gesture

    # =============================
    # Gesture Handlers
    # =============================

    def on_hand_close(self):
        self.fingertip_history = []
        self.curr_fingertip = None
        self.prev_fingertip = None
        self.fingertip_down = False

        if len(self.current_points) > 1:
            stroke_data = {
                "points": self.current_points[:],
                "color": list(self.current_color.keys())[0],
                "width": self.current_width,
                "eraser": self.eraser_mode,
                "abstracted": False
            }
            self.drawn_strokes.append(stroke_data)
        self.current_points = []

    def on_index_finger(self, x, y, frame):
        # Add the current point to the stroke
        self.current_points.append((float(x), float(y)))

        # Eraser mode
        if self.eraser_mode:
            if self.eraser_type == "Stroke":
                self.erase_all_stroke(x, y)
            else:
                self.erase_partial_at_point(x, y)

            if len(self.current_points) > 1:
                x1, y1 = self.current_points[-2]
                x2, y2 = self.current_points[-1]
                self.canvas.create_line(x1, y1, x2, y2, fill="white", width=20)
        else:
            # Normal drawing (pen mode)
            if len(self.current_points) > 1:
                x1, y1 = self.current_points[-2]
                x2, y2 = self.current_points[-1]
                self.canvas.create_line(x1, y1, x2, y2, fill=list(self.current_color.keys())[0],
                                        width=self.current_width)

        if not self.eraser_mode:
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
                color_bgr = constants.COLORS_OPTIONS[list(self.current_color.keys())[0]]
                cv2.polylines(frame, [pts], isClosed=False, color=color_bgr, thickness=self.current_width)
            cv2.circle(frame, (int(x), int(y)), 8, constants.COLORS_OPTIONS[list(self.current_color.keys())[0]], -1)
        else:
            cv2.circle(frame, (int(x), int(y)), 8, (128,128,128), -1)
        return frame

    def on_hand_3fingers(self, x, y):
        s_eraser_pressed = self.check_S_eraser_button_click(x, y)
        eraser_pressed = self.check_eraser_button_click(x, y)
        color_pressed = self.check_color_button_click(x, y)
        self.check_width_button_click(x, y)
        self.check_clear_button_click(x, y)
        #print(f"current color: {self.current_color}, current width: {self.current_width}, eraser mode: {self.eraser_mode},prev eraser mode: {self.prev_eraser_mode},last e-type:{self.last_eraser},eraser type: {self.eraser_type}")

        # Color Changed - Update Color Button
        if color_pressed and self.prev_color != self.current_color and self.prev_color is not None:
             self.create_buttons_overlay()
        # Color not changed but switched between eraser to color - Update Color Button
        elif color_pressed and self.prev_eraser_mode:
            self.create_buttons_overlay()

        elif self.eraser_mode and not self.prev_eraser_mode:
            self.create_buttons_overlay()
        elif (s_eraser_pressed or eraser_pressed) and (self.last_eraser!=self.eraser_type):
            self.create_buttons_overlay()
        if self.current_width != self.prev_width and self.prev_width is not None:
            self.create_buttons_overlay()

    def on_hand_open(self):
        if self.prev_gesture != self.current_gesture:
            print("Undo...")
            self.canvas.delete("all")

            # Mark one more stroke as abstracted
            for i in range(len(self.drawn_strokes) - 1, -1, -1):
                if not self.drawn_strokes[i].get("abstracted", False):
                    self.drawn_strokes[i]["abstracted"] = True
                    break

            # Re-draw
            self.draw_strokes_on_canvas()

    def on_hand_thumbsup(self):
        if self.prev_gesture != self.current_gesture:
            print("Smoothing the shapes...")
            self.canvas.delete("all")

            for stroke in self.drawn_strokes:
                points = stroke["points"]
                color = stroke["color"]
                width = stroke["width"]
                eraser_mode = stroke["eraser"]

                if eraser_mode:
                    self.draw_freehand(points, "white", width=20)
                else:
                    smoothed = chaikin_smoothing(points, iterations=2)
                    if stroke.get("abstracted", False):
                        self.draw_freehand(smoothed, color, width)
                    else:
                        shape, angle, aligned_edge = best_fit_shape(smoothed)
                        stroke["shape"] = shape
                        stroke["angle"] = angle
                        stroke["aligned_edge"] = aligned_edge

                        if shape == "line":
                            self.draw_line(smoothed, color, width)
                        elif shape == "circle":
                            self.draw_circle(smoothed, color, width)
                        elif shape == "ellipse":
                            self.draw_ellipse(smoothed, color, angle, width)
                        elif shape == "rectangle":
                            self.draw_rectangle(smoothed, color, angle, width)
                        elif shape == "triangle":
                            self.draw_triangle(smoothed, aligned_edge, color, angle, width)
                        else:
                            self.draw_freehand(smoothed, color, width)

    # ===========================
    # SECTION: Fingertip Methods
    # ===========================

    def stable_detect_fingertip(self, mask,history_size = constants.HISTORY_MAX_LENGTH):
        # Preprocess the mask to remove noise
        cleaned_mask = preprocess_mask(mask)

        # Detect new fingertip
        new_tip = detect_fingertip(cleaned_mask)

        # If new_tip is valid or we have no previous tip
        if new_tip is not None:
            # Check if we should accept this new tip based on jump threshold
            if is_valid_fingertip(new_tip, self.prev_fingertip):
                # Smooth it with EMA
                new_tip = (
                    smooth_fingertip(new_tip, self.prev_fingertip)
                    if self.prev_fingertip is not None
                    else new_tip
                )
                self.stability_counter = 0
            else:
                # The new tip is too far from the previous, might be a jump
                self.stability_counter += 1
                if self.stability_counter < constants.STABILITY_FRAMES:
                    # Use the old tip for now
                    new_tip = self.prev_fingertip
                else:
                    # If we've been unstable for too long, reset the counter
                    self.stability_counter = 0

            # Add the new tip (or the fallback) into the history
            if new_tip is not None:
                self.fingertip_history.append(new_tip)

                # If history too big, pop oldest
                if len(self.fingertip_history) > history_size:
                    self.fingertip_history.pop(0)

                # Take the median of recent points to further reduce jitter
                xs = [pt[0] for pt in self.fingertip_history]
                ys = [pt[1] for pt in self.fingertip_history]
                median_tip = (int(np.median(xs)), int(np.median(ys)))

                # Update current and previous fingertip
                self.curr_fingertip = median_tip
                self.prev_fingertip = median_tip
        else:
            # If we didn't detect a fingertip, keep the old position
            self.curr_fingertip = self.prev_fingertip

        return self.curr_fingertip
    # ========================
    # Eraser Methods
    # ========================

    def erase_all_stroke(self, x, y, threshold=10):
        for i in range(len(self.drawn_strokes) - 1, -1, -1):
            stroke = self.drawn_strokes[i]
            for (px, py) in stroke["points"]:
                dist = math.hypot(px - x, py - y)
                if dist < threshold:
                    self.drawn_strokes.pop(i)
                    self.draw_strokes_on_canvas()
                    break

    def erase_partial_at_point(self, x, y, radius=10):
        if self.mask[y, x] == 0:
            self.eraser_partial_mask[y - radius // 2:y + radius // 2,
                                     x - radius // 2:x + radius // 2] = 255

    # ============================
    # Drawing Functions
    # ============================

    def draw_strokes_on_canvas(self):
        for stroke in self.drawn_strokes:
            points = stroke["points"]
            color = stroke["color"]
            width = stroke["width"]
            eraser_mode = stroke["eraser"]
            smoothed = chaikin_smoothing(points, iterations=2)

            if eraser_mode:
                self.draw_freehand(points, "white", width=20)
            else:
                if stroke.get("abstracted", False):
                    self.draw_freehand(smoothed, color, width)
                else:
                    shape = stroke.get("shape", None)
                    angle = stroke.get("angle", 0)
                    aligned_edge = stroke.get("aligned_edge", "None")

                    if shape is None:
                        shape, angle, aligned_edge = best_fit_shape(smoothed)
                        stroke["shape"] = shape
                        stroke["angle"] = angle
                        stroke["aligned_edge"] = aligned_edge

                    if shape == "line":
                        self.draw_line(smoothed, color, width)
                    elif shape == "circle":
                        self.draw_circle(smoothed, color, width)
                    elif shape == "ellipse":
                        self.draw_ellipse(smoothed, color, angle, width)
                    elif shape == "rectangle":
                        self.draw_rectangle(smoothed, color, angle, width)
                    elif shape == "triangle":
                        self.draw_triangle(smoothed, aligned_edge, color, angle, width)
                    else:
                        self.draw_freehand(smoothed, color, width)

    def draw_freehand(self, points, color, width):
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def draw_line(self, points, color, width):
        x1, y1 = points[0]
        x2, y2 = points[-1]
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def draw_circle(self, points, color, width):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        size_width = max_x - min_x
        height = max_y - min_y
        diameter = max(size_width, height)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        left = cx - diameter / 2
        right = cx + diameter / 2
        top = cy - diameter / 2
        bottom = cy + diameter / 2
        self.canvas.create_oval(left, top, right, bottom, outline=color, fill="", width=width)

    def draw_ellipse(self, points, color, angle, width):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        theta_rad = math.radians(angle)
        rotated_points = []
        for point in points:
            rotated_points.append(rotate_point(point[0], point[1], -theta_rad, center))
        xe = [p[0] for p in rotated_points]
        ye = [p[1] for p in rotated_points]
        min_x, max_x = min(xe), max(xe)
        min_y, max_y = min(ye), max(ye)
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        a = (max_x - min_x) / 2
        b = (max_y - min_y) / 2

        ellipse_points = []
        for t in range(0, 360, 5):
            theta = math.radians(t)
            x = center[0] + a * math.cos(theta)
            y = center[1] + b * math.sin(theta)
            x_rot, y_rot = rotate_point(x, y, math.radians(angle), center)
            ellipse_points.append((x_rot, y_rot))

        self.canvas.create_polygon(ellipse_points, outline=color, fill="", width=width, smooth=True)

    def draw_rectangle(self, points, color, angle, width):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        theta_rad = math.radians(angle)
        rotated_points = []
        for point in points:
            rotated_points.append(rotate_point(point[0], point[1], -theta_rad, center))

        xr = [p[0] for p in rotated_points]
        yr = [p[1] for p in rotated_points]
        min_x, max_x = min(xr), max(xr)
        min_y, max_y = min(yr), max(yr)
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        rotated_corners = [rotate_point(x, y, math.radians(angle), center) for x, y in corners]
        self.canvas.create_polygon(rotated_corners, outline=color, fill="", width=width)

    def draw_triangle(self, points, aligned_edge, color, angle, width):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        center = (cx, cy)
        theta_rad = math.radians(angle)
        rotated_points = []
        for point in points:
            rotated_points.append(rotate_point(point[0], point[1], -theta_rad, center))

        xt = [p[0] for p in rotated_points]
        yt = [p[1] for p in rotated_points]
        min_x, max_x = min(xt), max(xt)
        min_y, max_y = min(yt), max(yt)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        center = (cx, cy)

        if aligned_edge == "Upper":
            v1 = (min_x, max_y)
            v2 = (max_x, max_y)
            for point in rotated_points:
                if point[1] == min_y:
                    v3 = (point[0], min_y)
        elif aligned_edge == "Right":
            v1 = (max_x, max_y)
            v2 = (max_x, min_y)
            for point in rotated_points:
                if point[0] == min_x:
                    v3 = (min_x, point[1])
        elif aligned_edge == "Lower":
            v1 = (min_x, min_y)
            v2 = (max_x, min_y)
            for point in rotated_points:
                if point[1] == max_y:
                    v3 = (point[0], max_y)
        elif aligned_edge == "Left":
            v1 = (min_x, min_y)
            v2 = (min_x, max_y)
            for point in rotated_points:
                if point[0] == max_x:
                    v3 = (max_x, point[1])

        v1_rot = rotate_point(v1[0], v1[1], theta_rad, center)
        v2_rot = rotate_point(v2[0], v2[1], theta_rad, center)
        v3_rot = rotate_point(v3[0], v3[1], theta_rad, center)

        self.canvas.create_line(v1_rot[0], v1_rot[1], v2_rot[0], v2_rot[1], fill=color, width=width)
        self.canvas.create_line(v1_rot[0], v1_rot[1], v3_rot[0], v3_rot[1], fill=color, width=width)
        self.canvas.create_line(v2_rot[0], v2_rot[1], v3_rot[0], v3_rot[1], fill=color, width=width)

    def draw_regular_polygon(self, points, color, sides=4):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        radius = min(width, height) / 2.0
        vertices = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append((x, y))
        for i in range(sides):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % sides]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=self.current_width)

    # =============================
    # Video / Main Loop
    # =============================

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            self.root.after(1, self.update_frame)
            return

        mask_frame = self.mask_func.proc_frame(frame)[0]
        gesture_name = self.svm_model.predict(mask_frame)
        gesture_name = self.stabilizer.update(gesture_name)
        self.update_gesture(gesture_name)

        frame_gui = frame.copy()
        frame_gui[self.mask != 0] = self.static_overlay[self.mask != 0] # draw Buttons

        if gesture_name == "up_thumb":
            gesture_name = "index_finger"
        print_gesture_on_frame(frame_gui, gesture_name)
        print_color_on_frame(frame_gui,list(self.current_color.keys())[0])

        # Handle Gestures
        if self.current_gesture == "close_hand":
            self.on_hand_close()
            self.on_hand_thumbsup() # Now close hand is thumb up, we can use 3fingers as close hand for now
        # THUMB UP AND INDEX FINGER ARE THE SAME
        #elif self.current_gesture == "up_thumb":
            #self.on_hand_close()
            #self.on_hand_thumbsup()
        elif self.current_gesture == "open_hand":
            self.on_hand_close()
            self.on_hand_open()
        # THUMB UP AND INDEX FINGER ARE THE SAME
        elif self.current_gesture == "index_finger" or self.current_gesture == "up_thumb":
            self.stable_detect_fingertip(mask_frame)
            if self.curr_fingertip:
                x, y = self.curr_fingertip[0], self.curr_fingertip[1]
                if x is not None and y is not None:
                    # if y < 70 and x > 5:
                    #     self.check_color_button_click(x, y)
                    #     self.check_eraser_button_click(x, y)
                    #     self.check_S_eraser_button_click(x, y)
                    #     self.check_clear_button_click(x, y)
                    # elif x < 70 and y > 80:
                    #     self.check_width_button_click(x, y)
                    if not self.fingertip_down:
                        self.fingertip_down = True
                        self.current_points = [(float(x), float(y))]
                    else:
                        frame_gui = self.on_index_finger(x, y, frame_gui)
        elif self.current_gesture == "three_fingers":
            #pos_x, pos_y = find_fingertip(mask_frame)
            self.stable_detect_fingertip(mask_frame,constants.HISTORY_MAX_LENGTH_3FINGERS)
            if self.curr_fingertip:
                pos_x,pos_y = self.curr_fingertip[0], self.curr_fingertip[1]
            cv2.circle(frame_gui, (int(pos_x), int(pos_y)), 10, (255, 255, 255), -1)
            self.on_hand_close()
            self.on_hand_3fingers(pos_x, pos_y)
        #else:
            #print("error - None gesture")

        # Draw completed strokes
        for stroke in self.drawn_strokes:
            if not stroke["eraser"]:
                pts = np.array(stroke["points"], dtype=np.int32).reshape((-1, 1, 2))
                color_bgr = constants.COLORS_OPTIONS[stroke["color"]]
                cv2.polylines(frame_gui, [pts], isClosed=False, color=color_bgr, thickness=stroke["width"])

        # Draw current stroke
        if self.current_gesture == "index_finger" and len(self.current_points) > 1:
            pts = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
            if not self.eraser_mode:
                color_bgr = constants.COLORS_OPTIONS[list(self.current_color.keys())[0]]
                cv2.polylines(frame_gui, [pts], isClosed=False, color=color_bgr, thickness=self.current_width)

        frame_gui[self.eraser_partial_mask != 0] = frame[self.eraser_partial_mask != 0] # eraser points
        cv2.imshow('GUI - DrawSomething Game', frame_gui)

        self.root.after(1, self.update_frame)

# ============================
# Run the Application (without word genarator app)
# ============================

def run():
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
