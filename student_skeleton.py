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
    # [핵심 트릭] 클래스를 건드리지 않고, 함수 스스로 이전 상태를 기억하도록 정적(Static) 변수처럼 활용
    if not hasattr(detect_monitor, "prev_corners"):
        detect_monitor.prev_corners = None
        detect_monitor.bad_count = 0

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    
    # 중앙 가중치를 위한 중심점 및 최대 거리 계산
    img_center = (img_w / 2.0, img_h / 2.0)
    max_dist = np.hypot(img_w, img_h) / 2.0

    blurred = cv2.bilateralFilter(image, 9, 75, 75)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1
    best_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        if img_area * 0.05 < area < img_area * 0.90:
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = rect
            rect_area = rw * rh
            
            if rect_area == 0:
                continue
                
            extent = area / rect_area
            aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
            
            if extent > 0.50 and 1.2 < aspect_ratio < 3.5:
                # [공간 방어] 화면 중앙에 가까울수록 가산점 (가장자리의 화려한 노이즈 배제)
                dist_to_center = np.hypot(cx - img_center[0], cy - img_center[1])
                center_weight = 1.0 - (dist_to_center / max_dist) * 0.3  # 0.7 ~ 1.0의 가중치
                
                score = area * extent * center_weight
                if score > best_score:
                    best_score = score
                    best_contour = c

    # 모니터 탐지에 실패했을 때의 처리
    if best_contour is None:
        if detect_monitor.prev_corners is not None:
            detect_monitor.bad_count += 1
            if detect_monitor.bad_count < 5:
                # 잠깐 안 보이는 거라면 이전 좌표로 버팀
                tl, tr, br, bl = detect_monitor.prev_corners
                return tuple(tl), tuple(tr), tuple(br), tuple(bl)
        # 완전히 잃어버렸다면 초기화
        detect_monitor.prev_corners = None
        return None, None, None, None

    # 4. 고무줄(Convex Hull) 씌우기
    screen_pts = None
    hull = cv2.convexHull(best_contour)
    hull_area = cv2.contourArea(hull)
    
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05]:
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            if cv2.contourArea(approx) / hull_area > 0.90: 
                screen_pts = approx.reshape(4, 2)
                break

    if screen_pts is None:
        rect_info = cv2.minAreaRect(best_contour)
        screen_pts = cv2.boxPoints(rect_info)

    # 좌표 정렬
    rect = np.zeros((4, 2), dtype="float32")
    s = screen_pts.sum(axis=1)
    rect[0] = screen_pts[np.argmin(s)]
    rect[2] = screen_pts[np.argmax(s)]
    diff = np.diff(screen_pts, axis=1)
    rect[1] = screen_pts[np.argmin(diff)]
    rect[3] = screen_pts[np.argmax(diff)]

    current_corners = rect

    # --- [시간 방어] 클래스 외부에서 동작하는 삑사리 및 Jitter 제어 ---
    if detect_monitor.prev_corners is None:
        detect_monitor.prev_corners = current_corners
        detect_monitor.bad_count = 0
    else:
        dist = np.max(np.linalg.norm(current_corners - detect_monitor.prev_corners, axis=1))
        
        if dist > 100.0:  # 100픽셀 이상 갑자기 튀면 노이즈(삑사리)로 간주
            detect_monitor.bad_count += 1
            if detect_monitor.bad_count < 5:
                current_corners = detect_monitor.prev_corners  # 이전 프레임 좌표 강제 유지
            else:
                detect_monitor.prev_corners = current_corners  # 화면 전환으로 인정
                detect_monitor.bad_count = 0
        else:
            # 부드러운 이동을 위한 지수 이동 평균 (EMA)
            alpha = 0.6
            current_corners = (1.0 - alpha) * detect_monitor.prev_corners + alpha * current_corners
            detect_monitor.prev_corners = current_corners
            detect_monitor.bad_count = 0

    tl, tr, br, bl = current_corners
    return tuple(tl), tuple(tr), tuple(br), tuple(bl)
    

def rectify_monitor(image, top_left, top_right, bottom_right, bottom_left):
    if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
        return None

    rect = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    (tl, tr, br, bl) = rect

    # 1. 실제 거리 기반 너비/높이 동적 계산 (회전 판단용)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 2. 16:9 표준 가로/세로 출력 크기 타깃 설정 (기본 해상도: 800 x 450)
    # 이미지 디테일을 더 살리고 싶다면 1280, 720 등으로 높여도 좋습니다.
    TARGET_WIDTH = 800
    TARGET_HEIGHT = 450

    # 3. 만약 세로가 가로보다 긴 상태(피벗 모니터)라면 타깃 비율을 일시적으로 세로형(450x800)으로 설정
    is_portrait = maxHeight > maxWidth
    if is_portrait:
        dst_w, dst_h = TARGET_HEIGHT, TARGET_WIDTH
    else:
        dst_w, dst_h = TARGET_WIDTH, TARGET_HEIGHT

    # 4. 투영 변환용 목적지(dst) 좌표계 매핑
    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]], dtype="float32")

    # 5. 원근 왜곡 보정 (Warp Perspective) 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    rectified = cv2.warpPerspective(image, M, (dst_w, dst_h))
    
    # 6. 세로형 모니터였던 경우, 이미지를 시계 방향으로 90도 회전시켜 최종 16:9(800x450) 규격으로 변환
    if is_portrait:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)
    
    return rectified

def detect_line(rectified):
    if rectified is None:
        return None

    h, w = rectified.shape[:2]
    
    # 1. ROI 설정 (베젤 그림자 배제)
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)
    roi = rectified[margin_y:h-margin_y, margin_x:w-margin_x]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 대비 극대화
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 2. [일반화 핵심 1] 엣지 민감도 상향
    # 빨간색(어두운 회색)과 검은색 사이의 미세한 차이도 엣지로 잡아내기 위해 상한선을 90으로 낮춤
    edges = cv2.Canny(blurred, 30, 90, apertureSize=3)
    
    # 윤곽선 찾기 (서로 엉겨붙지 않도록 날것의 엣지 그대로 사용)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # 깨끗한 도화지 준비
    clean_edges = np.zeros_like(edges)
    diag_len = np.hypot(w, h)
    
    for c in contours:
        # 3. [일반화 핵심 2] 덩어리를 꽉 감싸는 '최소 면적 사각형' 추출
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), angle = rect
        
        length = max(rw, rh)
        thickness = min(rw, rh)
        
        # 두께가 0인 1픽셀짜리 완벽한 선을 위한 방어 코드
        aspect_ratio = length / thickness if thickness > 0 else float('inf')
            
        # [궁극의 필터]
        # 조건 A: 길이가 화면 대각선의 5% 이상일 것 (자잘한 점, 먼지 제거)
        # 조건 B: 가로세로 비율이 1:4 이상일 것 (글씨, 별모양, 다각형, 캐릭터 윤곽선 전멸)
        if length > diag_len * 0.05 and aspect_ratio > 3.0:
            cv2.drawContours(clean_edges, [c], -1, 255, 1)

    # 4. 잡동사니가 멸종된 깨끗해진 도화지 위에서 허프 변환 수행
    min_line_len = int(diag_len * 0.15)
    
    lines = cv2.HoughLinesP(
        clean_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=40, 
        minLineLength=min_line_len, 
        maxLineGap=20  # 이제 노이즈가 없으므로 20픽셀 정도 끊겨있어도 안심하고 이어붙임
    )

    if lines is None:
        return None

    # 5. 가장 긴 선 찾기
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

    # 6. ROI 좌표 복원
    x1, y1, x2, y2 = longest_line
    return (x1 + margin_x, y1 + margin_y, x2 + margin_x, y2 + margin_y)
        
def calculate_angle(line):
    if line is None:
        return None

    x1, y1, x2, y2 = line
    
    # 선의 방향을 항상 '아래에서 위로' 강제
    if y1 < y2:
        x1, y1, x2, y2 = x2, y2, x1, y1
        
    dx = x2 - x1
    dy = y1 - y2  

    # 수직 0도, 왼쪽 +, 오른쪽 -
    angle_rad = np.arctan2(-dx, dy)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)

class LineDetector(Node):
    def __init__(self) -> None:
        super().__init__("line_detector_node")

        self.declare_parameter("topic_image", "/camera/camera/color/image_raw")
        self.declare_parameter("topic_student", "/student/angle")

        topic_image = str(self.get_parameter("topic_image").value)
        topic_student = str(self.get_parameter("topic_student").value)

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            topic_image,
            self.image_callback,
            10,
        )

        self.angle_pub = self.create_publisher(
            Float32,
            topic_student,
            10,
        )

        self.line_pub = self.create_publisher(
            Image,
            "/debug/line",
            10,
        )

        self.get_logger().info(
            f"Line detector started. Subscribing to {topic_image!r}, "
            f"publishing to {topic_student!r}."
        )

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc!r}")
            return

        # 1. Detect monitor.
        top_left, top_right, bottom_right, bottom_left = detect_monitor(image)
        if any(p is None for p in (top_left, top_right, bottom_right, bottom_left)):
            self.get_logger().warning("Monitor not detected.")
            return

        # 2. Rectify monitor.
        rectified = rectify_monitor(image, top_left, top_right, bottom_right, bottom_left)
        if rectified is None:
            self.get_logger().warning("Monitor not rectified.")
            return

        # 3. Detect line.
        line = detect_line(rectified)
        if line is None:
            self.get_logger().warning("Line not detected.")
            return
        self._debug_line(msg, rectified, line)

        # 4. Calculate and publish angle.
        angle = calculate_angle(line)
        if angle is None:
            self.get_logger().warning("Angle not calculated.")
            return

        angle_msg = Float32()
        angle_msg.data = float(angle)
        self.angle_pub.publish(angle_msg)

        self.get_logger().info(f"Line angle: {float(angle):.2f} deg")

    def _debug_line(self, msg, rectified, line) -> None:
        debug_line = rectified.copy()

        x1, y1, x2, y2 = line
        cv2.line(
            debug_line,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            6,
        )

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