#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


def detect_monitor(image):
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 40, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1
    best_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        if img_area * 0.05 < area < img_area * 0.85:
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            rect_area = rw * rh
            
            if rect_area == 0:
                continue
                
            extent = area / rect_area
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
            
            if extent > 0.65 and 1.2 < aspect_ratio < 3.0:
                score = area * extent 
                if score > best_score:
                    best_score = score
                    best_contour = c

    if best_contour is None:
        return None, None, None, None

    # 4. 고무줄(Convex Hull) 씌우기
    screen_pts = None
    hull = cv2.convexHull(best_contour)
    hull_area = cv2.contourArea(hull)
    
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        
        if len(approx) == 4:
            # [핵심 방어] 모서리가 파먹혔는지 면적으로 검사!
            # 4개의 점으로 만든 면적이 원래 고무줄(Hull) 면적보다 5% 이상 작아졌다면
            # (즉, 모서리가 잘려 나가서 사다리꼴이 되었다면) 이 4개 점을 가차 없이 버림
            approx_area = cv2.contourArea(approx)
            if approx_area / hull_area > 0.95: 
                screen_pts = approx.reshape(4, 2)
                break

    # 5. 모서리가 잘려 나갔거나 4점을 못 찾은 경우 (허공 꼭짓점 복원)
    # minAreaRect를 사용해 끊어진 선들을 가상으로 연장시켜 '진짜 꼭짓점'을 수학적으로 계산합니다.
    if screen_pts is None:
        rect_info = cv2.minAreaRect(best_contour)
        screen_pts = cv2.boxPoints(rect_info)

    # 6. 좌표 정렬 (절대 꼬이지 않는 합/차 정렬)
    rect = np.zeros((4, 2), dtype="float32")
    s = screen_pts.sum(axis=1)
    rect[0] = screen_pts[np.argmin(s)]
    rect[2] = screen_pts[np.argmax(s)]
    
    diff = np.diff(screen_pts, axis=1)
    rect[1] = screen_pts[np.argmin(diff)]
    rect[3] = screen_pts[np.argmax(diff)]

    tl, tr, br, bl = rect
    return tuple(tl), tuple(tr), tuple(br), tuple(bl)

    
def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
        return None

    rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    (tl, tr, br, bl) = rect

    # 16:9 강제 비율 제거, 실제 거리 기반 너비/높이 동적 계산
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    rectified = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return rectified


def detect_line(rectified):
    if rectified is None:
        return None

    h, w = rectified.shape[:2]
    
    # 1. 모니터 베젤(테두리) 그림자를 선으로 착각하지 않도록 상하좌우 5% 여백(ROI)을 잘라냅니다.
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    roi = rectified[margin_y:h-margin_y, margin_x:w-margin_x]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. 대비를 높이고 Canny 엣지로 선의 윤곽을 땁니다.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    
    # 3. 허프 변환 (HoughLinesP)으로 직선 추출
    # 화면 대각선 길이의 10% 이상 되는 선분만 찾도록 설정하여 자잘한 노이즈 무시
    min_len = int(np.hypot(w, h) * 0.1)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30, 
        minLineLength=min_len, 
        maxLineGap=20  # 살짝 끊어진 선도 하나의 선으로 이어줌
    )

    if lines is None:
        return None

    # 4. 찾은 선들 중에서 유클리드 거리(길이)가 가장 긴 선 하나만 선택
    longest_line = None
    max_length = -1

    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_length:
            max_length = length
            longest_line = (x1, y1, x2, y2)

    if longest_line is None:
        return None

    # 5. ROI 기준으로 찾은 좌표를 원본 평면화(Rectified) 이미지 좌표계로 원복
    x1, y1, x2, y2 = longest_line
    return (x1 + margin_x, y1 + margin_y, x2 + margin_x, y2 + margin_y)


def calculate_angle(line):
    if line is None:
        return None

    x1, y1, x2, y2 = line
    
    # 이미지 좌표계는 좌상단이 (0,0)이고 아래로 갈수록 y가 증가합니다.
    # 일반적인 데카르트 좌표계(수학/물리) 원점 기준에 맞춰 계산하기 위해 y축 방향을 뒤집어 줍니다.
    dy = y1 - y2  
    dx = x2 - x1
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # 로봇 제어를 위해 각도를 -90도 ~ 90도 사이로 정규화
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return float(angle_deg)


class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(Image, topic_image, self.image_callback, 10)
        self.angle_pub = self.create_publisher(Float32, topic_student, 10)
        self.line_pub = self.create_publisher(Image, "/debug/line", 10)
        self.corner_pub = self.create_publisher(Image, "/debug/corners", 10)
        
        # [새로 추가됨] 컴퓨터가 인식한 윤곽선을 그대로 띄워주는 토픽
        self.edge_pub = self.create_publisher(Image, "/debug/edges", 10)

        self.get_logger().info("Line detector started.")

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            return

        # --- [추가됨] 윤곽선(Edge) 이미지를 rqt_image_view로 바로 전송 ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        blurred = cv2.bilateralFilter(clahe.apply(gray), 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 150)
        
        edge_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
        edge_msg.header = msg.header
        self.edge_pub.publish(edge_msg)
        # ----------------------------------------------------------------

        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        self._debug_corners(msg, image, top_left, top_right, bottom_right, bottom_left)

        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            return

        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            return

        line = detect_line(rectified)
        if line is None:
            return
            
        self._debug_line(msg, rectified, line)

        angle = calculate_angle(line)
        if angle is None:
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

    def _debug_corners(self, msg, image, tl, tr, br, bl):
        debug_img = image.copy()
        if not any(p is None for p in (tl, tr, br, bl)):
            cv2.circle(debug_img, (int(tl[0]), int(tl[1])), 15, (0, 0, 255), -1)   
            cv2.circle(debug_img, (int(tr[0]), int(tr[1])), 15, (0, 255, 0), -1)   
            cv2.circle(debug_img, (int(br[0]), int(br[1])), 15, (255, 0, 0), -1)   
            cv2.circle(debug_img, (int(bl[0]), int(bl[1])), 15, (0, 255, 255), -1) 
            pts = np.array([tl, tr, br, bl], dtype=np.int32)
            cv2.polylines(debug_img, [pts], True, (255, 0, 255), 3)

        corner_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        corner_msg.header = msg.header
        self.corner_pub.publish(corner_msg)

    def _debug_line(self, msg, rectified, line) -> None:
        debug_line = rectified.copy()
        x1, y1, x2, y2 = line
        cv2.line(debug_line, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 6)
        debug_line_msg = self.bridge.cv2_to_imgmsg(debug_line, encoding="bgr8")
        debug_line_msg.header = msg.header
        self.line_pub.publish(debug_line_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
