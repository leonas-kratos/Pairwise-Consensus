# -*- coding: utf-8 -*-
"""
_robot_mixin.py — Phần điều khiển xe dùng chung cho LS / KF / EKF / PC-EKF
Chạy song song với filter bằng daemon thread.
"""

import threading
import time
import serial

# ── GPIO ──────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    GPIO = None
    _GPIO_AVAILABLE = False

STM_PORT = '/dev/serial/by-id/usb-STMicroelectronics_BLUEPILL_F103C8_CDC_in_FS_Mode_498E169E3634-if00'
STM_BAUD = 115200

IR_LEFT  = 27
IR_RIGHT = 17

_robot_lock    = threading.Lock()
_robot_running = False          # set True khi start_robot() gọi
_ser_stm       = None


def _setup_gpio():
    if not _GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IR_LEFT,  GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(IR_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


def _send_cmd(cmd: str):
    global _ser_stm
    if _ser_stm and _ser_stm.is_open:
        try:
            _ser_stm.write((cmd + '\n').encode())
        except Exception:
            pass


def _thread_ir():
    """Thread bám line bằng IR — gửi CMD về STM32."""
    prev_state = 'CLEAR'
    while True:
        with _robot_lock:
            if not _robot_running:
                break
        if _GPIO_AVAILABLE:
            ir_l = GPIO.input(IR_LEFT)
            ir_r = GPIO.input(IR_RIGHT)
            if ir_l == 1:
                state = 'LEFT'
            elif ir_r == 1:
                state = 'RIGHT'
            else:
                state = 'CLEAR'
        else:
            state = 'CLEAR'          # không có GPIO → giữ trạng thái rỗng

        if state != prev_state or state != 'CLEAR':
            _send_cmd(f'CMD,IR,{state}')
            prev_state = state

        time.sleep(0.05)


def start_robot():
    """
    Khởi động phần điều khiển xe.
    Gọi 1 lần ngay khi bắt đầu run() / demo().
    Trả về True nếu thành công, False nếu không có STM32 / GPIO.
    """
    global _robot_running, _ser_stm

    _setup_gpio()

    try:
        _ser_stm = serial.Serial(STM_PORT, STM_BAUD, timeout=0.1)
        time.sleep(2)                           # chờ STM32 boot
    except serial.SerialException as e:
        print(f"[ROBOT] Không mở được {STM_PORT}: {e}")
        print("[ROBOT] Chạy tiếp không điều khiển xe.")
        return False

    with _robot_lock:
        _robot_running = True

    t = threading.Thread(target=_thread_ir, daemon=True)
    t.start()

    _send_cmd('CMD,START')
    print(f"[ROBOT] Đã kết nối STM32 @ {STM_PORT}. Điều khiển xe bắt đầu.")
    return True


def stop_robot():
    """Dừng xe, đóng cổng STM32, cleanup GPIO."""
    global _robot_running, _ser_stm

    with _robot_lock:
        _robot_running = False

    _send_cmd('CMD,STOP')
    time.sleep(0.2)

    if _ser_stm:
        try:
            _ser_stm.close()
        except Exception:
            pass

    if _GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
        except Exception:
            pass

    print("[ROBOT] Đã dừng điều khiển xe.")
