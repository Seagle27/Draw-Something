"""

DrawingApp Class - with Camera/Video


"""

import tkinter as tk
import math
import cv2
import numpy as np
from  DrawSomething.segmentation import HandSegmentation
from DrawSomething import gesture_recognition as gest
from classifier import extract_hog_features, train
from DrawSomething import constants

# constants for fingertip detection
# Threshold for detecting a "jump" (in pixels)
JUMP_THRESHOLD = 50  # Adjust this value as needed
STABILITY_FRAMES = 12 #30  # Number of frames needed to accept a new position
EMA_ALPHA_MIN = 0.25  # Minimum smoothing factor (very smooth, slow response)
EMA_ALPHA_MAX = 0.9  # Maximum smoothing factor (fast response)
SPEED_THRESHOLD_LOW = 10   # Movement speed below this uses max smoothing
SPEED_THRESHOLD_HIGH = 30  # Movement speed above this uses min smoothing


def apply_brush_mask(brush_png_path, main_img, center_x, center_y, type="black",invert=True, threshold=128):
    """
    Applies a brush mask from a PNG image onto a main image.

    The function:
      - Loads the brush PNG image in grayscale.
      - Thresholds it to create a binary mask.
      - Resizes the mask to 60x60.
      - Overlays the mask onto the main_img at the given (center_x, center_y)
        position such that the pixels corresponding to the mask's white area (255)
        are set to black in the main image.

    Parameters:
      brush_png_path (str): Path to the brush/eraser PNG image.
      main_img (numpy.ndarray): Main image (cv2 image) where the mask will be applied.
      center_x (int): X-coordinate of the center for mask placement.
      center_y (int): Y-coordinate of the center for mask placement.
      threshold (int): Threshold value for binarization (default is 128).

    Returns:
      numpy.ndarray: The modified main image with the mask applied.
    """
    # 1. Load the brush image in grayscale
    brush_gray = cv2.imread(brush_png_path, cv2.IMREAD_GRAYSCALE)
    if brush_gray is None:
        raise ValueError(f"Could not load brush image from: {brush_png_path}")

    # 2. Create a binary mask by thresholding
    # Pixels >= threshold become 255, below become 0.
    _, mask = cv2.threshold(brush_gray, threshold, 255, cv2.THRESH_BINARY)

    # 3. Resize the mask to 60x60
    mask_60 = cv2.resize(mask, (45, 45), interpolation=cv2.INTER_AREA)

    # 4. Calculate the region of interest (ROI) in the main image
    mask_h, mask_w = mask_60.shape  # should be 60x60 - i changed to 45
    half_h, half_w = mask_h // 2, mask_w // 2

    # Compute the ROI coordinates while ensuring they stay within the main image bounds.
    y1 = max(0, center_y - half_h)
    y2 = min(main_img.shape[0], center_y + half_h)
    x1 = max(0, center_x - half_w)
    x2 = min(main_img.shape[1], center_x + half_w)

    # Adjust the mask region if the ROI is clipped (at the image boundaries)
    mask_y1 = half_h - (center_y - y1)
    mask_y2 = mask_y1 + (y2 - y1)
    mask_x1 = half_w - (center_x - x1)
    mask_x2 = mask_x1 + (x2 - x1)

    # 5. Apply the mask: set ROI pixels to black where mask value is 255.
    complementary_mask = cv2.bitwise_not(mask_60)
    roi = main_img[y1:y2, x1:x2]
    mask = mask_60
    if invert:
        mask = complementary_mask
    if type=="black":
        roi[mask[mask_y1:mask_y2, mask_x1:mask_x2] == 255] = (0, 0, 0)
    if type=="white":
        roi[mask[mask_y1:mask_y2, mask_x1:mask_x2] == 255] = (255, 255, 255)

    # 6. Return the modified image (still in cv2/NumPy format)
    return main_img

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
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width // 2}x{self.screen_height}+0+0")

        # ---------
        # Video
        # ---------
        # when using input from camera/video
        # self.cap = cv2.VideoCapture("records/triangle_mask.mp4")  # when working with video - or 0 for webcam
        self.cap = cv2.VideoCapture(0)
        self.mask_func = HandSegmentation(self.cap)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_height, self.frame_width = 480, 640
        self.svm_model = gest.SvmModel(constants.SVM_MODEL_PATH)
        self.stabilizer = gest.GestureStabilizer(constants.WIN_SIZE, constants.MIN_CHANGE_FRAME)

        self.fingertip_down = False  # State flag to know if "pressed" - not sure if needed
        self.root.after(1, self.update_frame)  # Start an update loop every 10ms
        #self.root.after_idle(self.update_frame)

        # ---------
        # Gestures
        # ---------

        self.current_gesture = "index_finger" # can be: index_finger,up_thumb,open_hand,close_hand,three_fingers
        self.prev_gesture = "close_hand"
        # ---------
        # Color Bar
        # ---------
        # colors for color bar - max colors:10

        # With less colors
        self.color_options = {
            "black": (0, 0, 0),
        #    "DarkRed": (0, 0, 139),
            "red": (0, 0, 255),
            "salmon": (122, 158, 227),
        #    "DarkGreen": (0, 100, 0),
            "green": (0, 255, 0),
        #    "GreenYellow": (47, 255, 173),
            "yellow": (0, 255, 255),
        #    "DodgerBlue": (30, 144, 255),
            "blue": (255, 0, 0)
        }
        # start position of color bar
        self.start_x = 30 #self.start_x = 40
        self.y = 30       #self.y = 40
        self.radius = 20  #self.radius = 30
        self.radius_of_selected = 30
        self.spacing = 80 #self.spacing = 120

        #self.brush_white_icon = "GUI_photos/mask_brush_white.npy"
        self.brush_black_icon = "GUI_photos/brush_black.png"
        self.eraser_black_icon = "GUI_photos/eraser_black.png"
        self.eraser_white_icon = "GUI_photos/eraser_white.png"

        # ---------
        # Width Bar
        # ---------
        self.width_options = [2, 4, 8, 12]
        # start position of width bar

        self.start_y = 110 #self.start_y = 160
        self.x = 30 #self.x = 40

        # Current drawing settings
        self.current_width = 2
        self.prev_width = None
        self.current_color = {"black":(0,0,0)} # NEED TO CHECK!
        self.prev_color = None
        self.eraser_mode = False
        self.last_eraser_mode = False

        # Store all completed strokes here
        # Each element: {"points": [...], "color": str, "eraser": bool}
        self.drawn_strokes = []
        self.drawn_strokes_gui = []
        self.prev_drawn_strokes = []
        self.all_strokes = []
        self.eraser_radius = 5 # the radius of the eraser (not the button)

        # Temporary list for the current stroke
        self.current_points = []
        # fingertip detection
        self.stability_counter = 0
        self.fingertip_history = []  # To store recent fingertip positions for median filtering
        self.history_max_length = 1  # Maximum history length
        self.curr_fingertip = None
        self.prev_fingertip = None

        # List of color buttons (circular) to display on canvas
        self.color_circles = []
        self.eraser_button = {}
        self.clear_button = {}  # will be next to eraser
        self.width_circles = []
        self.set_positions_dict()

        # Create main canvas
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.mask = None
        self.static_overlay = None
        # Create static overlays once (only if buttons do not change)
        self.create_buttons_overlay()

    # ---------------------
    # Buttons functions
    # ---------------------

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

    def check_clear_button_click(self, x, y):
        cx, cy = self.clear_button["center"]
        r = self.clear_button["radius"]
        if math.hypot(x - cx, y - cy) < r:
            self.clear_all_drawings()
            return True
        return False

    def check_width_button_click(self, x, y):
        for circle in self.width_circles:
            cx, cy = circle["center"]
            r = circle["radius"]
            dist = math.hypot(x - cx, y - cy)
            if dist <= r:
                self.set_width(circle["width"])
                return True
        return False

    # ---------------------
    # Set and update functions
    # ---------------------

    def set_color(self, color_key):
        self.prev_color = self.current_color
        color_rgb_value = self.color_options[color_key]
        self.current_color = {color_key: color_rgb_value}
        self.eraser_mode = False

    def set_width(self, width):
        self.prev_width = self.current_width
        self.current_width = width

    def clear_all_drawings(self):
        self.canvas.delete("all")
        self.drawn_strokes.clear()
        self.drawn_strokes_gui.clear()
        self.all_strokes.clear()

    def activate_eraser(self):
        self.last_eraser_mode = self.eraser_mode
        self.eraser_mode = True

    def update_gesture(self,gesture):
        self.prev_gesture = self.current_gesture
        self.current_gesture = gesture

    # ---------------------
    # Video and Hand functions
    # ---------------------

    def set_positions_dict(self):
        x = self.start_x
        colors = self.color_options

        for color_key,color_value in colors.items():
            self.color_circles.append({
                "color": color_key,
                "center": (x, self.y),
                "radius": self.radius
            })
            x += self.spacing

        #print(f"self.color_circles:{self.color_circles}")
        self.eraser_button = {
            "center": (x, self.y),
            "radius": self.radius
        }
        #print(f"self.eraser_button:{self.eraser_button}")
        x += self.spacing
        self.clear_button = {
            "center": (x, self.y),
            "radius": self.radius
        }
        #print(f"self.clear_button:{self.clear_button}")

        y = self.start_y
        x = self.x

        # Loop through each width option in the list
        for line_width in self.width_options:
            self.width_circles.append({
                "width": line_width,
                "center": (x, y),
                "radius": self.radius
            })
            y += self.spacing

        #print(f"self.width_circles:{self.width_circles}")

    def print_gesture_on_frame(self, frame, gesture="no gesture"):
        text = f"Detected Gesture: {gesture}"
        # height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX  # Use a valid OpenCV font constant
        font_scale = 0.65  # Define a font scale (adjust as needed)
        thickness = 1
        text_color = (255, 255, 255)
        cv2.putText(frame, text, (40, 440), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def print_color_on_frame(self, frame):
        text = f"Detected Color: {list(self.current_color.keys())[0]}"
        # height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX  # Use a valid OpenCV font constant
        font_scale = 0.65  # Define a font scale (adjust as needed)
        thickness = 1
        text_color = (255, 255, 255)
        cv2.putText(frame, text, (40, 460), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def create_buttons_overlay(self):
        overlay = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        # --- Draw Color Buttons (with eraser and clear) ---
        x = self.start_x
        for color_key, color_value in self.color_options.items():
            radius = self.radius
            if list(self.current_color.keys())[0] == color_key and not self.eraser_mode:
                radius = self.radius_of_selected
            cv2.circle(overlay, (x, self.y), radius, color_value, -1)
            cv2.circle(mask, (x, self.y), radius, 255, -1)

            if color_key == "black":
                overlay = apply_brush_mask(self.brush_black_icon, overlay, x, self.y, type="white", threshold=128)
            else:
                overlay = apply_brush_mask(self.brush_black_icon, overlay, x, self.y, type="black", threshold=128)
            x += self.spacing

        # Draw eraser button
        radius = self.radius if not self.eraser_mode else self.radius_of_selected
        gray = (128, 128, 128)
        cv2.circle(overlay, (x, self.y), radius, gray, -1)
        cv2.circle(mask, (x, self.y), radius, 255, -1)
        overlay = apply_brush_mask(self.eraser_black_icon, overlay, x, self.y, type="white", invert=False,
                                   threshold=128)
        overlay = apply_brush_mask(self.eraser_white_icon, overlay, x, self.y, type="black", invert=True, threshold=240)

        # Draw clear button
        x += self.spacing
        cv2.circle(overlay, (x, self.y), self.radius, gray, -1)
        cv2.circle(mask, (x, self.y), self.radius, 255, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)
        text3 = "clear"
        text_size3, _ = cv2.getTextSize(text3, font, font_scale, thickness)
        text_x3 = x - text_size3[0] // 2
        cv2.putText(overlay, text3, (text_x3, self.y + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # --- End of Color/Eraser/Clear Buttons ---

        # --- Draw Width Buttons ---
        # Set starting coordinates for width buttons
        y = self.start_y  # לדוגמה: 70
        x = self.x  # לדוגמה: 40
        for line_width in self.width_options:
            radius = self.radius
            if self.current_width == line_width:
                radius = self.radius_of_selected
            circle_color = (200, 200, 200)
            cv2.circle(overlay, (x, y), radius, circle_color, -1)
            cv2.circle(mask, (x, y), radius, 255, -1)

            # Draw a horizontal line inside the circle to represent the stroke width
            margin = 10  # מרווח מהקצה
            start_point = (x - self.radius + margin, y)
            end_point = (x + self.radius - margin, y)
            line_color = (0, 0, 0)
            cv2.line(overlay, start_point, end_point, line_color, line_width)

            y += self.spacing

        self.mask = mask
        self.static_overlay = overlay

    # In the update_frame method, always draw all previously completed strokes on frame_gui,
    # and draw the current stroke ONLY if the gesture is "index_finger."
    # ---------------------
    # Main Video loop
    # --------------------

    def update_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        #cv2.imshow('hand mask', mask_frame)
        frame_gui = frame.copy()

        mask_frame = self.mask_func.proc_frame(frame)
        gesture_name = self.svm_model.predict(mask_frame)
        gesture_name = self.stabilizer.update(gesture_name)

        self.update_gesture(gesture_name)

        frame_gui[self.mask != 0] = self.static_overlay[self.mask != 0]
        #frame_gui[self.mask] = self.static_overlay[self.mask]
        #frame_gui = cv2.add(frame_gui, self.static_overlay)

        # self.draw_color_buttons_on_frame(frame_gui)
        # self.draw_width_buttons_on_frame(frame_gui)
        self.print_gesture_on_frame(frame_gui, gesture_name)
        self.print_color_on_frame(frame_gui)

        if ret:
            if self.current_gesture == "close_hand":
                self.on_hand_close()
            elif self.current_gesture == "up_thumb":
                self.on_hand_close()
                #self.on_hand_thumbsup()
                self.on_hand_open()
            elif self.current_gesture == "open_hand":
                self.on_hand_close()
                self.on_hand_thumbsup()
                #self.on_hand_open()
            else:
                if self.current_gesture == "index_finger":
                    # x, y = self.find_fingertip(mask_frame)
                    # Alon's Algorithm
                    self.stable_detect_fingertip(mask_frame)
                    x, y = self.curr_fingertip[0], self.curr_fingertip[1]
                    if x is not None and y is not None:
                        #if y < 70 and x > 5:
                        #    self.check_color_button_click(x, y)
                        #    print(
                        #        f"1: current color: {self.current_color}, current width: {self.current_width}, eraser mode: {self.eraser_mode}")
                        #elif x < 70 and y > 80:
                        #    self.check_width_button_click(x,y)
                        #    print(
                        #        f"2: current color: {self.current_color}, current width: {self.current_width}, eraser mode: {self.eraser_mode}")
                        if not self.fingertip_down:
                            self.fingertip_down = True
                            self.current_points = [(float(x), float(y))]
                        else:
                            frame_gui = self.on_index_finger(x, y, frame_gui)
                elif self.current_gesture == "three_fingers":
                    pos_x , pos_y = self.find_fingertip(mask_frame)
                    self.on_hand_close()
                    self.on_hand_3fingers(pos_x, pos_y)
                else:
                    print("error - None gesture")

        # --- CHANGED BLOCK: Draw all *completed* strokes on frame_gui ---
        for stroke in self.drawn_strokes_gui:
            pts = np.array(stroke["points"], dtype=np.int32).reshape((-1, 1, 2))
            color_bgr = self.color_options[stroke["color"]]  # e.g., (0,0,255) for red
            cv2.polylines(frame_gui, [pts], isClosed=False, color=color_bgr, thickness=stroke["width"])


        # Draw ONLY the current stroke if gesture == "index_finger"
        if self.current_gesture == "index_finger" and len(self.current_points) > 1:
            pts = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
            if not self.eraser_mode:
                color_bgr = self.color_options[list(self.current_color.keys())[0]]
                cv2.polylines(frame_gui, [pts], isClosed=False, color=color_bgr, thickness=self.current_width)

        # # Set the Tkinter window geometry to half of the screen width:
        # frame_height, frame_width = self.frame_height, self.frame_width
        #
        # # Calculate a scaling factor (only scale down, do not scale up)
        # scale_factor = min(1, min(self.screen_width//2 / frame_width, self.screen_height / frame_height))
        # new_width = int(frame_width * scale_factor)
        # new_height = int(frame_height * scale_factor)
        #
        # # Resize only if needed
        # if scale_factor < 1:
        #     resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # else:
        #     resized_frame = frame
        #
        #
        # cv2.namedWindow("GUI - DrawSomething Game", cv2.WINDOW_NORMAL)
        #
        # cv2.resizeWindow("GUI - DrawSomething Game", resized_frame)
        # # Position it to the right of the Tkinter window
        # cv2.moveWindow("GUI - DrawSomething Game", self.screen_width // 2, 0)
        #
        cv2.imshow('GUI - DrawSomething Game', frame_gui)
        self.root.after(1, self.update_frame)

    # ---------------------
    # Gesture Event Handlers
    # ---------------------

    def on_hand_close(self):
        self.fingertip_history = []
        self.curr_fingertip = None
        self.prev_fingertip = None
        self.fingertip_down = False
        if len(self.current_points)>1:
            self.prev_drawn_strokes = self.drawn_strokes
            stroke_data = {
                "points": self.current_points[:],
                "color": list(self.current_color.keys())[0],
                "width": self.current_width,
                "eraser": self.eraser_mode,
                "abstracted": False
            }
            self.drawn_strokes.append(stroke_data)
            stroke_data_gui = {
                "points": self.current_points[:],
                "color": list(self.current_color.keys())[0],
                "width": self.current_width,
            }
            self.drawn_strokes_gui.append(stroke_data_gui)
        #print(len(self.drawn_strokes))
        self.current_points = []

    #    if we are in eraser_mode, call 'erase_strokes_at'
    #    instead of drawing new lines on frame_gui.
    #    We still update your canvas with a white line for local Tkinter erasing,
    #    but for the frame_gui (camera feed), we do geometry-based removal.

    def on_index_finger(self, x, y, frame):
        # Add the current point to the stroke
        self.current_points.append((float(x), float(y)))

        # If eraser mode, remove from existing strokes instead of drawing a new stroke on frame_gui.
        if self.eraser_mode:
            # Erase from the stored stroke data near current fingertip
            self.erase_strokes_at(x, y, self.eraser_radius)

            # Also erase on the Tkinter canvas by drawing a thick white line
            # so you visually see it being erased in the canvas widget:
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

        # Return the frame unchanged if in eraser mode. In pen mode you do the temporary stroke.
        if not self.eraser_mode:
            if len(self.current_points) > 1:
                pts = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
                color_bgr = self.color_options[list(self.current_color.keys())[0]]
                cv2.polylines(frame, [pts], isClosed=False, color=color_bgr, thickness=self.current_width)
            cv2.circle(frame, (int(x), int(y)), 8, self.color_options[list(self.current_color.keys())[0]], -1) # (0,255,255)

        return frame

    def finalize_stroke(self):
        """
        Call this method when the finger is lifted or the stroke is finished.
        It saves the current stroke and resets it for the next drawing.
        """
        if self.current_points:
            self.all_strokes.append(self.current_points.copy())
            self.current_points = []

    def on_hand_3fingers(self,x,y):
        self.check_eraser_button_click(x, y)
        self.check_color_button_click(x,y)
        self.check_width_button_click(x,y)
        self.check_clear_button_click(x,y)
        # color changed
        if self.eraser_mode == False and self.prev_color != self.current_color and self.prev_color != None:
            self.create_buttons_overlay()
        # eraser mode changed
        elif self.eraser_mode != self.last_eraser_mode:
            self.create_buttons_overlay()
        # width changed
        if self.current_width != self.prev_width and self.prev_width != None:
            self.create_buttons_overlay()

        #print(f"current color: {self.current_color}, current width: {self.current_width}, eraser mode: {self.eraser_mode}")

    def on_hand_open(self):
        if self.prev_gesture != self.current_gesture:
            print("Undo...")
            self.canvas.delete("all")

            # Mark one more stroke as abstracted
            for i in range(len(self.drawn_strokes) - 1, -1, -1):
                if not self.drawn_strokes[i].get("abstracted", False):
                    self.drawn_strokes[i]["abstracted"] = True
                    break  # mark only one stroke

            # Now re-draw all strokes
            for stroke in self.drawn_strokes:
                points = stroke["points"]
                color = stroke["color"]
                width = stroke["width"]
                eraser_mode = stroke["eraser"]
                smoothed = self.chaikin_smoothing(points, iterations=2)

                if eraser_mode:
                    self.draw_freehand(points, "white", width=20)
                else:
                    if stroke.get("abstracted", False):
                        # Draw as freehand if abstract
                        self.draw_freehand(smoothed, color,width)
                    else:
                        # Use pre-computed shape details if they exist
                        shape = stroke.get("shape", None)
                        angle = stroke.get("angle", 0)
                        aligned_edge = stroke.get("aligned_edge", "None")

                        # If shape was never computed (e.g., stroke added after last Ctrl+S),
                        # then compute now and store it.
                        if shape is None:
                            shape, angle, aligned_edge = self.best_fit_shape(smoothed)
                            stroke["shape"] = shape
                            stroke["angle"] = angle
                            stroke["aligned_edge"] = aligned_edge

                        # Draw using the stored shape data
                        if shape == "line":
                            self.draw_line(smoothed, color,width)
                        elif shape == "circle":
                            self.draw_circle(smoothed, color,width)
                        elif shape == "ellipse":
                            self.draw_ellipse(smoothed, color, angle,width)
                        elif shape == "rectangle":
                            self.draw_rectangle(smoothed, color, angle,width)
                        elif shape == "triangle":
                            self.draw_triangle(smoothed, aligned_edge, color, angle,width)
                        else:
                            self.draw_freehand(smoothed, color,width)

    def on_hand_thumbsup(self):
        #print(f"self.prev_drawn_strokes:{self.prev_drawn_strokes},self.drawn_strokes:{self.drawn_strokes}")
        #if self.prev_drawn_strokes != self.drawn_strokes:
        print("Smoothing the shapes...")
        #print(f"self.drawn_strokes: {self.drawn_strokes}")

        self.canvas.delete("all")
        for stroke in self.drawn_strokes:
            points = stroke["points"]
            color = stroke["color"]
            width = stroke["width"]
            eraser_mode = stroke["eraser"]

            if eraser_mode:
                self.draw_freehand(points, "white", width=20)
            else:
                # Smooth the stroke.
                smoothed = self.chaikin_smoothing(points, iterations=2)
                if stroke.get("abstracted", False):
                    # Already marked as abstract → draw freehand.
                    self.draw_freehand(smoothed, color)
                else:
                    # Process with best-fit detection.
                    shape, angle, aligned_edge = self.best_fit_shape(smoothed)
                    # Added
                    stroke["shape"] = shape
                    stroke["angle"] = angle
                    stroke["aligned_edge"] = aligned_edge

                    if shape == "line":
                        self.draw_line(smoothed, color,width)
                    elif shape == "circle":
                        self.draw_circle(smoothed, color,width)
                    elif shape == "ellipse":
                        self.draw_ellipse(smoothed, color, angle,width)
                    elif shape == "rectangle":
                        self.draw_rectangle(smoothed, color, angle,width)
                    elif shape == "triangle":
                        self.draw_triangle(smoothed, aligned_edge, color, angle,width)
                    else:
                        self.draw_freehand(smoothed, color,width)

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
    # Helper functions
    # ---------------------

    def segment_points(self, threshold=100):
        new_strokes = []
        for stroke in self.drawn_strokes_gui:
            pts = stroke["points"]
            if pts:
                segments = []
                current_segment = [pts[0]]
                for i in range(1, len(pts)):
                    if math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) > threshold:
                        segments.append(current_segment)
                        current_segment = [pts[i]]
                    else:
                        current_segment.append(pts[i])
                segments.append(current_segment)
                for seg in segments:
                    if seg:  # Ensure the segment is not empty
                        new_stroke = {
                            "points": seg,
                            "color": stroke["color"],
                            "width": stroke["width"]
                        }
                        new_strokes.append(new_stroke)
        self.drawn_strokes_gui = new_strokes

    # --------------------
    # fingertip detection
    # --------------------
    # simple function
    def find_fingertip(self, frame):
        # If frame (mask_frame) is already grayscale, no need to convert
        gray = frame  # Just use it directly

        # Threshold to ensure binary (maybe it's already binary, but okay to redo)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Choose the largest contour (assuming it's the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Example: take the topmost point of that contour as "fingertip"
        topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        return topmost[0], topmost[1]

    def detect_fingertip(self, mask):
        """Detects the most probable fingertip from a binary hand mask using convex hull."""

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None  # No hand detected

        # Get the largest contour (assuming it's the hand)
        hand_contour = max(contours, key=cv2.contourArea)

        # Compute convex hull (this ensures a smooth outer boundary)
        hull = cv2.convexHull(hand_contour)

        # Compute hand center (palm center)
        M = cv2.moments(hand_contour)
        if M["m00"] == 0:
            return None  # Avoid division by zero
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Identify potential fingertips: Convex hull points that are FAR from the palm center
        max_distance = 0
        fingertip = None

        for point in hull[:, 0]:  # Iterate over hull points
            x, y = point
            if y > cy: continue
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Condition: Fingertip is a point with the **maximum distance from the palm center**
            if distance > max_distance:
                max_distance = distance
                fingertip = (x, y)

        return fingertip  # Returns (x, y) coordinates of the detected fingertip

    def is_valid_fingertip(self, new_tip, prev_tip, threshold=JUMP_THRESHOLD):
        """Checks if the new fingertip is within a reasonable distance of the previous one."""
        if prev_tip is None:
            return True  # Always accept the first detected point
        distance = np.linalg.norm(np.array(new_tip) - np.array(prev_tip))
        return distance < threshold  # Accept only if within threshold

    def dynamic_ema_alpha(self, speed):
        """Adjusts EMA_ALPHA dynamically based on movement speed."""
        if speed < SPEED_THRESHOLD_LOW:
            return EMA_ALPHA_MIN  # Very smooth for slow movement
        elif speed > SPEED_THRESHOLD_HIGH:
            return EMA_ALPHA_MAX  # Fast response for rapid movement
        else:
            # Linearly interpolate between min and max alpha
            return EMA_ALPHA_MIN + (EMA_ALPHA_MAX - EMA_ALPHA_MIN) * (
                        (speed - SPEED_THRESHOLD_LOW) / (SPEED_THRESHOLD_HIGH - SPEED_THRESHOLD_LOW))

    def smooth_fingertip(self, curr_tip, prev_tip):
        """Smooths fingertip position using an Exponential Moving Average (EMA)."""
        speed = np.linalg.norm(np.array(prev_tip) - np.array(curr_tip))
        if prev_tip is None:
            return curr_tip

        alpha = self.dynamic_ema_alpha(speed)
        return (
            int(alpha * curr_tip[0] + (1 - alpha) * prev_tip[0]),
            int(alpha * curr_tip[1] + (1 - alpha) * prev_tip[1])
        )

    def stable_detect_fingertip(self, mask):
        x,y = self.find_fingertip(mask)
        new_tip = (x,y)
        #new_tip = self.detect_fingertip(mask)
        if new_tip is not None:
            #print("Detected new tip:", new_tip)
            if self.prev_fingertip is None or self.is_valid_fingertip(new_tip, self.prev_fingertip):
                new_tip = self.smooth_fingertip(new_tip, self.prev_fingertip) if self.prev_fingertip else new_tip
                self.stability_counter = 0
                #print("Valid tip. Smoothed tip:", new_tip)
            else:
                self.stability_counter += 1
                #print("Big jump detected. Stability counter:", self.stability_counter)
                if self.stability_counter < STABILITY_FRAMES:
                    new_tip = self.prev_fingertip
                    #print("Using previous tip due to instability.")
                else:
                    self.stability_counter = 0
                    #print("Accepting new tip after sustained instability.")

            self.fingertip_history.append(new_tip)
            if len(self.fingertip_history) > self.history_max_length:
                self.fingertip_history.pop(0)
            xs = [pt[0] for pt in self.fingertip_history]
            ys = [pt[1] for pt in self.fingertip_history]
            median_tip = (int(np.median(xs)), int(np.median(ys)))
            self.curr_fingertip = median_tip
            self.prev_fingertip = median_tip
            #print("Median filtered tip:", median_tip)
        else:
            #print("No fingertip detected, maintaining previous tip.")
            self.curr_fingertip = self.prev_fingertip
    # --------------------
    # helper method to remove points from strokes near an (x, y) position.
    # --------------------

    def erase_strokes_at(self, x, y, radius):
        """Removes from each stroke any points lying within 'radius' of (x, y)
        and updates the stroke's segments using segment_points."""
        for stroke in self.drawn_strokes_gui:
            new_points = []
            for (px, py) in stroke["points"]:
                dist = math.hypot(px - x, py - y)
                if dist > radius:
                    new_points.append((px, py))
            stroke["points"] = new_points
            # Update segments after erasing points
            self.segment_points(threshold=100)

    # --------------------
    # points functions
    # --------------------

    def rotate_point(self, x, y, theta, center=(0, 0)):
        """
        Rotates a point (x, y) around a given center by an angle theta (in radians).

        :param x: The x-coordinate of the point to rotate.
        :param y: The y-coordinate of the point to rotate.
        :param theta: The angle of rotation in radians. A positive value rotates the point counterclockwise.
        :param center: A tuple (a, b) representing the center of rotation. Defaults to (0, 0) if not specified.
        :return: A tuple (x_new, y_new) representing the new coordinates of the rotated point.
        """
        # Extract the coordinates of the center of rotation.
        a, b = center

        # Translate the point so that the center of rotation becomes the origin.
        # This is achieved by subtracting the center's coordinates from the point's coordinates.
        x_rel = x - a
        y_rel = y - b

        # Apply the rotation transformation using the standard rotation formulas:
        #   x_rot = x_rel * cos(theta) - y_rel * sin(theta)
        #   y_rot = x_rel * sin(theta) + y_rel * cos(theta)
        # These formulas compute the coordinates of the rotated point relative to the origin.
        x_rot = x_rel * math.cos(theta) - y_rel * math.sin(theta)
        y_rot = x_rel * math.sin(theta) + y_rel * math.cos(theta)

        # Translate the rotated coordinates back to the original coordinate system by adding the center's coordinates.
        x_new = x_rot + a
        y_new = y_rot + b

        # Return the new coordinates of the rotated point.
        return x_new, y_new

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

    # -----------------------------------
    # Initial estimate for the shape type
    # -----------------------------------

    def detect_shape_by_gradient_orientations_from_points(self, points,
                                                          low_threshold=100,
                                                          peak_threshold_ratio=0.2):
        """
        1) Create a small raster image from the provided 'points'.
        2) Applies a small Gaussian blur to reduce noise.
        3) Computes the Sobel derivatives in X and Y directions.
        4) Calculates the gradient magnitude and orientation for each pixel.
        5) Keeps only pixels above a certain gradient magnitude threshold (low_threshold).
        6) Builds a histogram of gradient orientations (0-180 degrees).
        7) Detects the number of prominent peaks in this orientation histogram.
        8) Infers the shape based on the number of peaks:
           - 3 major peaks -> triangle
           - 2 or 4 major peaks -> square / rectangle
           - No distinct peaks or uniform distribution -> circle / curved shape
        9) Displays the raster image and histogram using matplotlib.

        Returns one of:
          "Triangle", "Square/Rectangle", "Circle (or close to circle)", or "Unknown".
        """

        if len(points) < 2:
            #print("1. Unknown (not enough points)")
            return "Unknown"

        # 1) Determine bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))

        width = max_x - min_x + 20  # small padding
        height = max_y - min_y + 20
        if width < 2 or height < 2:
            #print("2. Unknown (bounding box too small)")
            return "Unknown"

        # Create a blank white image (numpy array)
        img = np.ones((height, width), dtype=np.uint8) * 255  # grayscale, white

        # Shift points so that (min_x - 10, min_y - 10) becomes (0,0) in our image
        shift_x = min_x - 10
        shift_y = min_y - 10

        # Draw the stroke in black on the image
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            cv2.line(
                img,
                (int(x1 - shift_x), int(y1 - shift_y)),
                (int(x2 - shift_x), int(y2 - shift_y)),
                color=(0,),  # black
                thickness=4
            )

        # 2) Apply a small Gaussian blur to reduce noise
        #    ksize can be 3 or 5 depending on how much smoothing
        blurred = cv2.GaussianBlur(img, (7, 7), 10)

        # 3) Compute Sobel derivatives in X and Y directions
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # 4) Calculate gradient magnitude and orientation
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        angle = np.arctan2(sobel_y, sobel_x)
        angle_deg = np.degrees(angle)

        # 5) Keep only pixels above low_threshold in magnitude
        valid_mask = magnitude > low_threshold
        valid_angles = angle_deg[valid_mask]
        if len(valid_angles) == 0:
            #print("3. Unknown (no edges above threshold)")
            return "abstract"

        # 6) Build a histogram of orientations (0..180) in 36 bins
        bins = 36
        hist, bin_edges = np.histogram(valid_angles, bins=bins, range=(-180, 180))
        max_val = np.max(hist)
        if max_val == 0:
            #print("4. Unknown (empty histogram)")
            return "abstract"

        # 7) Detect the number of prominent peaks
        peak_threshold = peak_threshold_ratio * max_val
        peaks = []
        for i in range(1, bins - 1):
            # check if hist[i] is a local maximum
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= peak_threshold:
                peaks.append(i)

        # Now explicitly check the last bin
        # (but you have to decide what "neighbors" means at the boundary).
        i = bins - 1
        if hist[i] > hist[i - 1] and hist[i] >= peak_threshold:
            peaks.append(i)

        num_peaks = len(peaks)

        # 8) Infer shape based on number of peaks
        if num_peaks == 6:
            shape_name = "Triangle Shape"
        elif num_peaks == 4:
            shape_name = "Square Shape"
        # elif num_peaks == 10:
        #    shape_name = "Polygon5"
        elif num_peaks == 5:
            shape_name = "Square Shape"
        elif num_peaks >= 9:
            shape_name = "Circular Shape"
        elif num_peaks in [7, 8]:
            shape_name = "Unknown Shape"  # can be triangle,circular,square
        else:
            shape_name = "abstract"

        # For Debugging:

        # 9) Display the raster image and the histogram - for debugging
        # plt.figure(figsize=(12, 6))
        #
        # # Show the raster image
        # plt.subplot(1, 3, 1)
        # plt.imshow(img, cmap='gray')
        # plt.title('Raster Image (After Drawing)')
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 2)
        # plt.imshow(blurred, cmap='gray')
        # plt.title('blurred Image (After Drawing)')
        # plt.axis('off')
        #
        # # Show the histogram
        # plt.subplot(1, 3, 3)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
        # plt.title('Orientation Histogram')
        # plt.xlabel('Angle (degrees)')
        # plt.ylabel('Count')
        #
        # plt.tight_layout()
        # plt.show()

        #print(f"Detected shape: {shape_name}, with {num_peaks} major peaks in orientation.")
        return shape_name

    # ---------------------
    # Best-Fit Shape
    # ---------------------

    def best_fit_shape(self, points):
        """
        Attempts to match several shapes (Line, Circle, Ellipse, Rectangle, Triangle)
        and returns the shape with the lowest average error (if it is below a certain threshold).
        Otherwise, returns "abstract".
        """
        # Define a threshold such that if the average error is too high, the shape is not selected
        # we can use different thresholds to different shapes.
        THRESHOLD_abstract = 15.0
        # THRESHOLD = 40.0
        THRESHOLD = 15.0

        #print("--- Detect Shape By Gradient Orientations From Points ---")
        smoothed = self.chaikin_smoothing(points, iterations=2)
        aprrox_shape = self.detect_shape_by_gradient_orientations_from_points(smoothed)
        #print(f"aprrox_shape: {aprrox_shape}")

        # Set initial variable
        best_shape = "abstract"
        best_error = float("inf")
        best_count = 0
        best_angle = 0
        aligned_edge = "None"

        # Compute the center of the shape using the bounding box of the points.
        xs = [p[0] for p in points]  # x coordinates
        ys = [p[1] for p in points]  # y coordinates
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Center of the shape
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        center = (cx, cy)

        # Iterate over rotation angles from 0° to 90° in steps of 5°.
        for angle in range(0, 91, 5):
            rotated_points = []
            theta_rad = math.radians(angle)  # Convert degrees to radians
            # print()

            # Rotate all points by -theta_rad around the computed center.
            for point in points:
                rotated_point = self.rotate_point(point[0], point[1], -theta_rad, center)
                rotated_points.append(rotated_point)
            # print(f"First Point:{points[0]}, The rotate point:{rotated_points[0]}")

            # Find the Relevant Candidate Shape and their errors
            if aprrox_shape == "Triangle Shape":
                candidates_rotates = {"triangle": self.count_close_point_triangle(rotated_points)}
            elif aprrox_shape == "Square Shape":
                candidates_rotates = {"rectangle": self.error_rectangle(rotated_points)}
            elif aprrox_shape == "Circular Shape":
                candidates_rotates = {
                    "circle": self.error_circle(rotated_points),
                    "ellipse": self.error_ellipse(rotated_points)
                }
            elif aprrox_shape == "Unknown Shape":
                candidates_rotates = {
                    "line": self.error_line(rotated_points),
                    "circle": self.error_circle(rotated_points),
                    "ellipse": self.error_ellipse(rotated_points),
                    "triangle1": self.error_triangle(rotated_points),
                    "rectangle": self.error_rectangle(rotated_points)
                }
            else:
                candidates_rotates = {
                    "line": self.error_line(rotated_points),
                    "ellipse": self.error_ellipse(rotated_points)
                    # ,"rectangle": self.error_rectangle(rotated_points)
                }
            # print(f"\nAngle: {angle}°,Shape name: {aprrox_shape},Candidates:{candidates_rotates}")

            # Update the best match if a candidate has a lower error.
            for shape_name, err in candidates_rotates.items():
                # print(shape_name, err)
                if shape_name != "triangle":
                    if err < best_error:
                        best_error = err
                        best_shape = shape_name
                        best_angle = angle
                        aligned_edge = "None"

                # Act different for triangle:
                else:
                    count, side = err
                    if count > best_count and side != "None":
                        best_count = count
                        best_shape = shape_name
                        best_angle = angle
                        aligned_edge = side

            # print(f"best_count:{best_count},best error:{best_error},best shape:{best_shape},best angle:{best_angle}")

            # Break the loop early
            if best_error < 7:
                break
        #print(f"best_count:{best_count},best error:{best_error},best shape:{best_shape},best angle:{best_angle}")
        if aprrox_shape == "Triangle Shape":
            return best_shape, best_angle, aligned_edge

        elif aprrox_shape == "Unknown Shape" and best_shape == "triangle":
            for angle in range(0, 91, 5):
                rotated_points = []
                theta_rad = math.radians(angle)  # Convert degrees to radians

                # Rotate all points by -theta_rad around the computed center.
                for point in points:
                    rotated_point = self.rotate_point(point[0], point[1], -theta_rad, center)
                    rotated_points.append(rotated_point)
                    count, side = self.count_close_point_triangle(rotated_points)
                    if count > best_count and side != "None":
                        best_count = count
                        best_shape = shape_name
                        best_angle = angle
                        aligned_edge = side
            return best_shape, best_angle, aligned_edge

        # for using different thresholds
        elif ((aprrox_shape == "abstract" and best_error < THRESHOLD_abstract) or
              (aprrox_shape == "Unknown Shape" and best_error < THRESHOLD_abstract) or
              (aprrox_shape in ["Circular Shape", "Square Shape"] and best_error < THRESHOLD)):

            return best_shape, best_angle, aligned_edge

        # If no shape met the threshold, return "abstract" with the last evaluated angle.
        #print("will draw abstract")
        return "abstract", 0, "None"

    # -- Error metrics for shapes --

    def error_line(self, points):
        """
        Calculates the average perpendicular distance from each point to the line defined by the first and last point.
        If there are fewer than two points, or if the first and last point are almost identical,
        it returns a large error value (9999) indicating an invalid or degenerate line.
        """
        # Check if there are at least two points to define a line
        if len(points) < 2:
            return 9999  # Not enough points to form a line, so return a high error value

        # Unpack the first and last points from the list
        x1, y1 = points[0]  # The first point (starting point of the line)
        x2, y2 = points[-1]  # The last point (ending point of the line)

        # Compute the differences in the x and y coordinates between the two points
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the length of the line segment using the Pythagorean theorem
        length = math.hypot(dx, dy)

        # If the length is extremely small, the points are nearly identical,
        # so return a high error value as the line is not well-defined
        if length < 1e-5:
            return 9999  # The line is too short to be meaningful

        # Initialize a variable to accumulate the sum of perpendicular distances
        total_dist = 0

        # Iterate over each point to calculate its perpendicular distance to the line
        for (x, y) in points:
            # Calculate the perpendicular distance using the line formula:
            # distance = |dy * x - dx * y + (x2 * y1 - y2 * x1)| / length
            # This is derived from the standard formula for the distance from a point to a line.
            dist = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length
            # Accumulate the computed distance
            total_dist += dist

        # Return the average distance by dividing the total distance by the number of points
        return total_dist / len(points)

    def error_circle(self, points):
        """
        Computes a circle (center + radius) based on the bounding box of the points,
        then calculates the average deviation of each point's distance from the center relative to the average radius.
        """
        # Ensure there are enough points to define a circle (at least 3 points are needed)
        if len(points) < 3:
            return 9999  # Not enough points to form a circle; return a high error value

        # Separate the x and y coordinates from the points
        xs = [p[0] for p in points]  # Extract all x-coordinates
        ys = [p[1] for p in points]  # Extract all y-coordinates

        # Find the minimum and maximum values of x and y to determine the bounding box
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate the center of the circle as the center of the bounding box
        cx = (min_x + max_x) / 2.0  # x-coordinate of the center
        cy = (min_y + max_y) / 2.0  # y-coordinate of the center

        # Compute the distance from the center to each point using the Euclidean distance formula (hypotenuse)
        dists = [math.hypot(x - cx, y - cy) for (x, y) in points]

        # In the unlikely event that no distances are calculated, return a high error value
        if not dists:
            return 9999

        # Calculate the average radius as the mean of the distances from the center to each point
        r_avg = sum(dists) / len(dists)

        # Compute the total error as the sum of absolute differences between each point's distance and the average radius
        total_error = sum(abs(d - r_avg) for d in dists)

        # Return the average error per point, indicating how much the points deviate from a perfect circle
        return total_error / len(points)

    def error_ellipse(self, points):
        """
        Fit an "ellipse" based on the bounding box of the points:
          - The center is calculated as (cx, cy) = ((min_x + max_x)/2, (min_y + max_y)/2)
          - The radii are defined as rx = (max_x - min_x)/2 and ry = (max_y - min_y)/2.
        Then, for each point, it evaluates how closely the point fits the ellipse equation:
          (dx^2)/(rx^2) + (dy^2)/(ry^2) ≈ 1,
        where (dx, dy) is the vector from the center to the point.
        The function returns an error metric which is the average absolute deviation from 1,
        scaled by a factor (here, 25). A lower value indicates that the points better fit an ellipse.
        """
        # Check if there are enough points to define an ellipse (need at least 3)
        if len(points) < 3:
            return 9999  # Not enough points; return a high error value

        # Separate the x and y coordinates from all points
        xs = [p[0] for p in points]  # List of all x coordinates
        ys = [p[1] for p in points]  # List of all y coordinates

        # Determine the bounds of the points by finding the minimum and maximum x and y values
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Calculate the width and height of the bounding box
        width = max_x - min_x
        height = max_y - min_y

        # If the bounding box is too small in either dimension, the ellipse is ill-defined
        if width < 1 or height < 1:
            return 9999  # Return a high error value for degenerate cases

        # Compute the center of the bounding box, which serves as the center of the ellipse
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        # The radii of the ellipse are half the width and height of the bounding box
        rx = width / 2
        ry = height / 2

        # Initialize the total error accumulator
        total_error = 0

        # For each point, calculate its normalized squared distance from the center
        for (x, y) in points:
            dx = x - cx  # Horizontal distance from the center
            dy = y - cy  # Vertical distance from the center
            # Compute the normalized value based on the ellipse equation:
            # Ideally, for a perfect ellipse, (dx^2)/(rx^2) + (dy^2)/(ry^2) should equal 1.
            val = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry)
            # Accumulate the absolute deviation from the ideal value of 1
            total_error += abs(val - 1.0)

        # The commented-out return statements show alternative scaling factors for the error.
        # For instance:
        #   return 25 * (total_error / len(points))
        #   return 35 * (total_error / len(points)) + 1
        # The final chosen error metric multiplies the average deviation by 25.

        # Return the scaled average error over all points
        return 25 * (total_error / len(points))

    def error_rectangle(self, points):
        """
        Improves the error calculation for a rectangle:
        Defines an axis-aligned rectangle based on the bounding box,
        then computes for each point the minimum distance to one of the four sides.
        The error is the average of these distances.
        """
        # Ensure there are at least two points to form a rectangle; otherwise, return a high error.
        if len(points) < 2:
            return 9999  # Not enough data points to define a rectangle

        # Extract the x and y coordinates from each point
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Determine the minimum and maximum x and y values to compute the bounding box
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Check if the bounding box is sufficiently large in both dimensions
        if (max_x - min_x) < 1 or (max_y - min_y) < 1:
            return 9999  # If the bounding box is too small, consider it an invalid rectangle

        # Define the four edges of the rectangle using the bounding box coordinates.
        # Each edge is defined by its two endpoints (x1, y1) and (x2, y2).
        edges = [
            (min_x, min_y, max_x, min_y),  # Top edge: from left to right along the top
            (max_x, min_y, max_x, max_y),  # Right edge: from top to bottom on the right side
            (max_x, max_y, min_x, max_y),  # Bottom edge: from right to left along the bottom
            (min_x, max_y, min_x, min_y)  # Left edge: from bottom to top on the left side
        ]

        # Initialize a variable to accumulate the total minimum distance from all points to the rectangle edges.
        total_dist = 0

        # For each point, compute the minimum distance to any of the four rectangle edges.
        for (px, py) in points:
            # List to store distances from the current point to each edge.
            dists = []
            # Calculate distance from the point to each edge using a helper function.
            for (x1, y1, x2, y2) in edges:
                d = self.line_segment_distance(px, py, x1, y1, x2, y2)
                dists.append(d)
            # The error for the current point is the smallest distance to any edge.
            min_dist_for_point = min(dists)
            # Accumulate the minimum distance error.
            total_dist += min_dist_for_point

        # Compute the average error over all points.
        avg_dist = total_dist / len(points)

        # return avg_dist

        # Scale and offset the average error to adjust the metric.
        # The final error is scaled by 1.2 and then increased by 1.
        return 1.2 * avg_dist + 1

    # -------------------------------------
    # Triangle Function - Different approach
    # --------------------------------------
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

        # return 0.3*(total_dist / len(points))

        return (0.5 * (total_dist / len(points)) + 1)

    def count_close_point_triangle(self, points, epsilon_ratio=0.01):
        # Compute Bounding Box
        # Extract all x and y coordinates from the list of points
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        # Determine the bounding box of all points by finding the minimum and maximum x and y coordinates
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute the width and height of the bounding box
        width = max_x - min_x
        height = max_y - min_y
        # Define a small threshold epsilon based on the smaller dimension of the bounding box.
        # This threshold will be used to decide if a point is "close enough" to an edge.
        epsilon = epsilon_ratio * (min(width, height))

        # print(f"min_x={min_x},max_x={max_x},min_y={min_y}, max_y={max_y}")
        count_points_on_edge1 = 0  # edge 1 - upper edge ((min_x,max_y)->(max_x,max_y))
        count_points_on_edge2 = 0  # edge 2 - right edge ((max_x,max_y)->(max_x,min_y))
        count_points_on_edge3 = 0  # edge 3 - lower edge ((min_x,min_y)->(max_x,min_y))
        count_points_on_edge4 = 0  # edge 4 - left edgt ((min_x,min_y)->(min_x,max_y))

        # Loop through each point to check if it is close to any of the bounding box edges
        for point in points:
            # Check if the point is near the upper edge by comparing its y-coordinate with max_y.
            if abs(point[1] - max_y) < epsilon:
                # print(f"point near edge1 with distance y={point[1]}\n")
                count_points_on_edge1 += 1  # Increment counter for the upper edge

            # Check if the point is near the right edge by comparing its x-coordinate with max_x.
            if abs(point[0] - max_x) < epsilon:
                # print(f"point near edge2 with distance x={point[0]}\n")
                count_points_on_edge2 += 1  # Increment counter for the right edge

            # Check if the point is near the lower edge by comparing its y-coordinate with min_y.
            if abs(point[1] - min_y) < epsilon:
                # print(f"point near edge3 with distance y={point[1]}\n")
                count_points_on_edge3 += 1  # Increment counter for the lower edge

            # Check if the point is near the left edge by comparing its x-coordinate with min_x.
            if abs(point[0] - min_x) < epsilon:
                # print(f"point near edge4 with distance x={point[0]}\n")
                count_points_on_edge4 += 1  # Increment counter for the left edge

        # Create a list containing the counts for easier analysis of which edge has the most points
        counter_arr = [count_points_on_edge1, count_points_on_edge2, count_points_on_edge3, count_points_on_edge4]

        # Identify the maximum number of points aligned with any edge
        selected_count = max(counter_arr)
        # Get the index of the edge with the most points (0: Upper, 1: Right, 2: Lower, 3: Left)
        selected_edge = counter_arr.index(selected_count)

        # Check if the maximum count is at least 3 and return the corresponding edge labe
        if selected_count >= 3 and selected_edge == 0:
            # print(f"aligned by Upper with {selected_count} points.")
            return selected_count, "Upper"
        elif selected_count >= 3 and selected_edge == 1:
            # print(f"aligned by Right with {selected_count} points.")
            return selected_count, "Right"
        elif selected_count >= 3 and selected_edge == 2:
            # print(f"aligned by Lower with {selected_count} points.")
            return selected_count, "Lower"
        elif selected_count >= 3 and selected_edge == 3:
            # print(f"aligned by Left with {selected_count} points.")
            return selected_count, "Left"
        else:
            # If no edge has at least 3 points close to it, return the count and "None" as the alignment.
            return selected_count, "None"

    # -------------------
    # Drawing Functions
    # -------------------

    def draw_freehand(self, points, color,width):
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

    def draw_line(self, points, color,width):
        # Unpack the first and last points from the list
        x1, y1 = points[0]  # The first point (starting point of the line)
        x2, y2 = points[-1]  # The last point (ending point of the line)
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width= width)
        # Draw on the OpenCV video frame
        # Note: Ensure that 'color' is a BGR tuple for OpenCV (e.g., (255, 0, 0) for blue)
        # need to add frame
        #cv2.line(frame, (x1, y1), (x2, y2), self.color_options[color], width)

    def draw_circle(self, points, color,width):
        # Find bounding box for computing the center
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

        self.canvas.create_oval(left, top, right, bottom, outline=color, fill="", width=width)
        #cv2.circle(frame, center, radius, self.color_options[color], self.current_width)

    # allow rotations
    def draw_ellipse(self, points, color, angle,width):
        """Draws an ellipse by rotating its boundary box."""
        # Find bounding box for computing the center
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute the center
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)

        # Aligning the ellipse
        theta_rad = math.radians(angle)
        rotated_points = []
        for point in points:
            rotated_points.append(self.rotate_point(point[0], point[1], -theta_rad, center))

        # Find the new bounding box
        xe = [p[0] for p in rotated_points]
        ye = [p[1] for p in rotated_points]
        min_x, max_x = min(xe), max(xe)
        min_y, max_y = min(ye), max(ye)

        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        a = (max_x - min_x) / 2  # Semi-major axis
        b = (max_y - min_y) / 2  # Semi-minor axis

        # Generate points along the ellipse before rotation
        ellipse_points = []
        for t in range(0, 360, 5):  # Sample points every 5 degrees
            theta = math.radians(t)
            x = center[0] + a * math.cos(theta)
            y = center[1] + b * math.sin(theta)

            # Rotate the sampled point
            x_rot, y_rot = self.rotate_point(x, y, math.radians(angle), center)
            ellipse_points.append((x_rot, y_rot))

        # Draw the rotated ellipse as a polygon approximation
        self.canvas.create_polygon(ellipse_points, outline=color, fill="", width=width, smooth=True)
        # Draw on the OpenCV frame
        # OpenCV doesn't support smoothing like Tkinter, so this will be a basic polygon
        # Prepare points for OpenCV: convert to a NumPy array of integers
        #pts = np.array(ellipse_points, dtype=np.int32).reshape((-1, 1, 2))
        #cv2.polylines(frame, ellipse_points, isClosed=True, color=self.color_options[color], thickness=self.current_width)

    # allow rotations
    def draw_rectangle(self, points, color, angle,width):
        """Draws a rotated rectangle by rotating its four corner points."""
        # Find bounding box for computing the center
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute the center
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)

        # Aligning the rectangle
        theta_rad = math.radians(angle)
        rotated_points = []
        for point in points:
            rotated_points.append(self.rotate_point(point[0], point[1], -theta_rad, center))

        # Find the new bounding box
        xr = [p[0] for p in rotated_points]
        yr = [p[1] for p in rotated_points]
        min_x, max_x = min(xr), max(xr)
        min_y, max_y = min(yr), max(yr)

        # Define the four corners
        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        # Rotate each corner
        rotated_corners = [self.rotate_point(x, y, math.radians(angle), center) for x, y in corners]

        # Draw polygon for rotated rectangle
        self.canvas.create_polygon(rotated_corners, outline=color, fill="", width=width)
        # Prepare points for OpenCV: convert to a NumPy array of integers

        # Draw on the OpenCV frame
        #pts = np.array(rotated_corners, dtype=np.int32).reshape((-1, 1, 2))
        #cv2.polylines(self.frame, [pts], isClosed=True, color=self.color_options[color], thickness=self.current_width)

    def draw_triangle(self, points, aligned_edge, color, angle,width):
        """
        Draws a rotated triangle by rotating its three vertices.
        :param points: The bounding box points for reference.
        :param color: Color of the triangle.
        :param angle: Rotation angle in degrees.
        """
        #print(f"angle={angle}")
        # Find bounding box for computing the center
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Compute triangle center
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        center = (cx, cy)

        # Aligning the triangle
        theta_rad = math.radians(angle)
        rotated_points = []
        for point in points:
            rotated_points.append(self.rotate_point(point[0], point[1], -theta_rad, center))

        # Find the new bounding box
        xt = [p[0] for p in rotated_points]
        yt = [p[1] for p in rotated_points]
        min_x, max_x = min(xt), max(xt)
        min_y, max_y = min(yt), max(yt)

        # Compute triangle center
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        center = (cx, cy)

        # find 3 points of triangle
        #print(aligned_edge)
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

        # Rotate the vertices around the center
        v1_rot = self.rotate_point(v1[0], v1[1], theta_rad, center)  # theta_rad
        v2_rot = self.rotate_point(v2[0], v2[1], theta_rad, center)
        v3_rot = self.rotate_point(v3[0], v3[1], theta_rad, center)

        # Draw the rotated triangle
        self.canvas.create_line(v1_rot[0], v1_rot[1], v2_rot[0], v2_rot[1], fill=color, width=width)
        self.canvas.create_line(v1_rot[0], v1_rot[1], v3_rot[0], v3_rot[1], fill=color, width=width)
        self.canvas.create_line(v2_rot[0], v2_rot[1], v3_rot[0], v3_rot[1], fill=color, width=width)

        # Draw on the OpenCV frame
        #vertices = [v1_rot,v2_rot,v3_rot]
        #pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        #cv2.polylines(frame, [pts], isClosed=True, color=self.color_options[color], thickness=self.current_width)


        # Draw the rotated triangle
        # self.canvas.create_polygon(v1_rot, v2_rot, v3_rot, outline=color, fill="", width=2)

    def draw_regular_polygon(self, points, color, sides=4):
        """
        Draw a regular polygon with 'sides' vertices inside the bounding box.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        radius = min(width, height) / 2.0

        # Calculate each vertex
        vertices = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides - math.pi / 2  # start from top
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            vertices.append((x, y))

        # Draw the polygon edges
        for i in range(sides):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % sides]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=self.current_width)

def run():
    root = tk.Tk()

    #root.withdraw()  # Hide the Tkinter window
    app = DrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()