## To visualize and compare the performance of the teacher and student models
import cv2
import numpy as np

class Visualizer: 
    ##Inspired by OpenAI Gym's rendering utilities, 
    ## as well as DeepMind visualization tools for RL agents. 
    ## Renders side-by-side visual comparison of Teacher vs Student agents
    ## WITH Real Time HUD overlays, Q-Values and latency metrics.
    def __init__(self, frame_size=(320, 320)):
        self.frame_size = frame_size

    def _prepare_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        ### Converting observation matrices into std RGB renderable frames
        if raw_frame.ndim == 1:
            if raw_frame.size == 100800:
                raw_frame = raw_frame.reshape(210,160,3)
            else:
                dim = int(np.sqrt(raw_frame.size)) ##Handling flattened/single channel inputs
                raw_frame = raw_frame.reshape(dim,dim)
        if raw_frame.ndim == 2:
            raw_frame = cv2.cvtColor((raw_frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif raw_frame.dtype != np.uint8:
                raw_frame = (raw_frame * 255).astype(np.uint8) if raw_frame.max() <=1.0 else raw_frame.astype(np.uint8)

        return cv2.resize(raw_frame, self.frame_size, interpolation=cv2.INTER_NEAREST) ## Resize to standard frame size for consistent visualization

    def draw_hud(self, frame: np.ndarray, title: str, action: int, q_value: float, latency_ms: float, color=(0,255,0)) ->np.ndarray:
        ## Draw a heads-up display(HUD) overlay on the frame with action, q-value, and latency metrics.
        #Overlays telemetry info directly onto render frames
        canvas = frame.copy()

        ##Header
        cv2.rectangle(canvas, (0,0), (self.frame_size[0], 35), (20,20,20), -1)
        cv2.putText(canvas, title, (10,23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        ## Telemetry info
        cv2.rectangle(canvas, (0,self.frame_size[1]-50), (self.frame_size[0], self.frame_size[1]), (20,20,20), -1)
        cv2.putText(canvas, f"Action: {action} | Q:{q_value:.2f}", (10, self.frame_size[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Latency: {latency_ms:.2f}ms", (10, self.frame_size[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

        return canvas

    def render_side_by_side(self, teacher_frame: np.ndarray, student_frame: np.ndarray, t_action: int, t_q: float, t_lat:float, s_action: int, s_q: float, s_lat: float):
        t_img = self._prepare_frame(teacher_frame)
        s_img = self._prepare_frame(student_frame)

        t_hud = self.draw_hud(t_img, "Teacher(NatureCNN - 1.68M)", t_action, t_q, t_lat, color=(0,255,255))
        s_hud = self.draw_hud(s_img, "Student(Lightweight - 223K)", s_action, s_q, s_lat, color=(0,255,0))
        combined = np.hstack((t_hud, s_hud)) # Horintal side by side concat
        cv2.imshow("Teacher vs Student Telemetry", combined)

        cv2.waitKey(1)  # Wait for 1 ms to allow for real-time rendering