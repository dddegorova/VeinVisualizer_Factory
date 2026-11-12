"""Управление тремя осями CNC Shield V3 через GRBL 1.1 и pyserial.

Обычное соответствие GRBL на Arduino Uno:
 X: STEP D2, DIR D5; Y: STEP D3, DIR D6; Z: STEP D4, DIR D7.
 ВАЖНО: двигатель Z должен быть подключен к разъёму Z (не A),
 либо разъём A должен аппаратно клонировать Z через перемычки.

Перед командой zero каретка должна находиться в реальном (0, 0, 0).
 Программные ограничения не заменяют концевые выключатели!
"""

import math
import time

import serial


PORT = "COM6"       # Windows: COM6; macOS: /dev/cu.usbmodem... или /dev/cu.usbserial...
BAUD_RATE = 115200

LIMITS = {"X": 45.0, "Y": 190.0, "Z": 110.0}  # мм
STEP_MM = 4.0
FEED_RATE = 300  # мм/мин: осторожная скорость для первого теста


class GRBLController:
    def __init__(self, port=PORT, baud_rate=BAUD_RATE):
        self.serial = serial.Serial(port, baud_rate, timeout=0.2, write_timeout=2)
        self.target = None  # известная целевая позиция; None до установки нуля
        self.paused = False

        # Открытие COM-порта может перезапускать Arduino.
        time.sleep(2)
        self.serial.reset_input_buffer()
        print(f"Подключено к {port}.")
        print("Сначала проверьте состояние командой status и установите ноль командой zero.")

    def send_command(self, command, timeout=5):
        """Отправляет команду и ждёт ok/error, но НЕ завершения движения."""
        self.serial.write((command + "\n").encode("ascii"))
        end = time.monotonic() + timeout
        received = []

        while time.monotonic() < end:
            line = self.serial.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            received.append(line)
            if line == "ok":
                return received
            if line.startswith("error:") or line.startswith("ALARM:"):
                raise RuntimeError("GRBL: " + "; ".join(received))

        raise TimeoutError(f"GRBL не ответил на команду {command!r}: {received}")

    def status(self):
        """Запрос состояния GRBL: Idle, Run, Hold, Alarm и координаты."""
        self.serial.write(b"?")
        end = time.monotonic() + 3
        while time.monotonic() < end:
            line = self.serial.readline().decode("ascii", errors="replace").strip()
            if line.startswith("<") and line.endswith(">"):
                return line
            if line.startswith("ALARM:"):
                raise RuntimeError(line)
        raise TimeoutError("Нет ответа на запрос статуса GRBL")

    def machine_state(self):
        return self.status()[1:].split("|", 1)[0].rstrip("> ").split(":", 1)[0]

    def unlock(self):
        """Снимает тревогу, но НЕ определяет положение каретки."""
        print("ВНИМАНИЕ: $X только снимает блокировку; координаты после него неизвестны.")
        print("Проверьте, что каретка не упирается в ограничители.")
        if input("Для разблокировки напишите YES: ").strip() != "YES":
            print("Отменено.")
            return
        print("\n".join(self.send_command("$X")))
        self.target = None
        self.paused = False
        print("Теперь установите реальную исходную позицию и выполните zero.")

    def set_zero(self):
        """Назначает текущую физическую позицию координатами X0 Y0 Z0."""
        state = self.machine_state()
        if state != "Idle":
            raise RuntimeError(
                f"Нельзя ставить ноль в состоянии {state}. "
                "Дождитесь остановки; при Alarm проверьте причину тревоги."
            )
        print("Устанавливайте ноль ТОЛЬКО в физическом начале рабочего диапазона")
        print("по всем трём осям: от этого положения допустимы X+, Y+, Z+.")
        if input("Каретка действительно в (0, 0, 0)? Напишите YES: ").strip() != "YES":
            print("Установка нуля отменена.")
            return

        self.send_command("G21 G90 G54")  # мм, абсолютные координаты, G54
        self.send_command("G92.1")        # удалить старое временное смещение G92
        self.send_command("G92 X0 Y0 Z0")
        self.target = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        self.paused = False
        print("Ноль установлен: X=0, Y=0, Z=0 мм.")

    def move_to(self, x=None, y=None, z=None, feed_rate=FEED_RATE):
        if self.target is None:
            raise RuntimeError("Сначала поставьте каретку в исходную точку и введите zero.")
        if self.paused:
            raise RuntimeError("Движение на паузе. Используйте c или reset и новый zero.")

        values = {"X": x, "Y": y, "Z": z}
        goal = self.target.copy()
        for axis, value in values.items():
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or not (0 <= value <= LIMITS[axis]):
                raise ValueError(f"Ось {axis}: допустимый диапазон 0..{LIMITS[axis]:g} мм")
            goal[axis] = value

        if goal == self.target:
            print("Каретка уже направлена в эту точку.")
            return
        if self.machine_state() in ("Alarm", "Hold", "Door"):
            raise RuntimeError("GRBL заблокирован или находится на паузе. Проверьте status.")

        gcode = (f"G1 X{goal['X']:.3f} Y{goal['Y']:.3f} "
                 f"Z{goal['Z']:.3f} F{feed_rate:g}")
        self.send_command(gcode)
        self.target = goal   # задание принято; физически движение может ещё идти
        print(f"Цель: X={goal['X']:.3f}, Y={goal['Y']:.3f}, Z={goal['Z']:.3f} мм")

    def move_relative(self, axis, distance, feed_rate=FEED_RATE):
        axis = axis.upper()
        if axis not in LIMITS:
            raise ValueError("Ось должна быть X, Y или Z")
        if self.target is None:
            raise RuntimeError("Сначала выполните zero")
        goal = self.target[axis] + float(distance)
        self.move_to(**{axis.lower(): goal}, feed_rate=feed_rate)

    def stop(self):
        """Плавная остановка GRBL. Это НЕ аварийное отключение питания."""
        self.serial.write(b"!")
        self.paused = True
        print("Пауза (feed hold). Для продолжения введите c.")

    def resume(self):
        if not self.paused:
            print("Пауза не включена.")
            return
        self.serial.write(b"~")
        self.paused = False
        print("Продолжаем ранее заданное движение.")

    def show_settings(self):
        print("\n".join(self.send_command("$$")))

    def configure_travel(self):
        """Записывает величины ходов в энергонезависимую память GRBL."""
        print("Будут записаны максимальные ходы GRBL: X=45, Y=190, Z=110 мм.")
        print("Это не включает защиту $20: она требует корректного homing и концевиков.")
        if input("Продолжить? Напишите YES: ").strip() != "YES":
            print("Отменено.")
            return
        for axis, setting in (("X", 130), ("Y", 131), ("Z", 132)):
            self.send_command(f"${setting}={LIMITS[axis]:g}")
        print("Настройки $130, $131, $132 сохранены.")

    def close(self):
        if self.serial.is_open:
            # Не бросаем неожиданно двигатель на ходу при обычном выходе.
            try:
                if self.machine_state() == "Run":
                    self.stop()
            except Exception:
                pass
            self.serial.close()
            print("COM-порт закрыт.")


def show_help():
    print("""
Команды (все координаты — в миллиметрах):
  zero            задать (0,0,0) в текущей ФИЗИЧЕСКОЙ точке
  status          текущее состояние GRBL и реальные координаты
  x 20            перейти на X=20
  y 100           перейти на Y=100
  z 50            перейти на Z=50
  xyz 20 100 50   перейти в точку (20, 100, 50)
  r / l           X +4 / -4
  u / d           Y +4 / -4
  f / b           Z +4 / -4
  dx 2            X +2 (для dy, dz — аналогично; можно отрицательно)
  20              как x 20 (совместимость со старым кодом)
  s               плавная остановка/пауза
  c               продолжить
  unlock          снять ALARM через $X (НЕ устанавливает ноль)
  limits          записать $130=45, $131=190, $132=110
  settings        вывести настройки GRBL ($$)
  q               выход

Границы: 0 <= X <= 45, 0 <= Y <= 190, 0 <= Z <= 110.
Перед первым движением необходимо выполнить zero.
""")


def main():
    grbl = None
    try:
        grbl = GRBLController()

        # Ускорение осей (мм/с²)
        grbl.send_command("$120=20")  # X
        grbl.send_command("$121=20")  # Y
        grbl.send_command("$122=20")  # Z

        show_help()
        while True:
            cmd = input("Введите команду: ").strip().lower().replace(",", ".")
            if not cmd:
                continue
            parts = cmd.split()
            op = parts[0]

            try:
                if op == "q":
                    break
                elif op in ("help", "h", "?"):
                    show_help()
                elif op == "status":
                    print(grbl.status())
                    print("Последняя заданная цель:", grbl.target)
                elif op == "zero":
                    grbl.set_zero()
                elif op == "unlock":
                    grbl.unlock()
                elif op == "limits":
                    grbl.configure_travel()
                elif op == "settings":
                    grbl.show_settings()
                elif op == "s":
                    grbl.stop()
                elif op == "c":
                    grbl.resume()
                elif len(parts) == 1 and op in ("r", "l", "u", "d", "f", "b"):
                    moves = {"r": ("X", STEP_MM), "l": ("X", -STEP_MM),
                             "u": ("Y", STEP_MM), "d": ("Y", -STEP_MM),
                             "f": ("Z", STEP_MM), "b": ("Z", -STEP_MM)}
                    axis, distance = moves[op]
                    grbl.move_relative(axis, distance)
                elif op in ("dx", "dy", "dz") and len(parts) == 2:
                    grbl.move_relative(op[1], float(parts[1]))
                elif op in ("x", "y", "z") and len(parts) == 2:
                    grbl.move_to(**{op: float(parts[1])})
                elif op == "xyz" and len(parts) == 4:
                    grbl.move_to(float(parts[1]), float(parts[2]), float(parts[3]))
                elif len(parts) == 1:
                    grbl.move_to(x=float(op))
                else:
                    print("Неизвестная команда. Введите help.")
            except (RuntimeError, ValueError, TimeoutError) as error:
                print("Ошибка:", error)

    except serial.SerialException as error:
        print("Ошибка порта:", error)
        print("Проверьте PORT и то, что Arduino IDE закрыла монитор порта.")
    except KeyboardInterrupt:
        print("\nПрерывание. Останавливаем каретку.")
        if grbl is not None:
            try:
                grbl.stop()
            except Exception:
                pass
    finally:
        if grbl is not None:
            grbl.close()


if __name__ == "__main__":
    main()
