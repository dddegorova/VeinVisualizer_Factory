import serial
import time


class GRBLController:
    def __init__(self, port='/dev/cu.usbmodem12345', baud_rate=115200):
        try:
            self.serial = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)
            self.wakeup()
            print(f"✅ Подключено к {port}")
        except serial.SerialException as e:
            print(f"❌ Ошибка подключения: {e}")
            print("💡 Проверьте, правильно ли указан порт (см. в Arduino IDE Tools → Port)")
            raise

    def send_command(self, command):
        self.serial.write(f"{command}\n".encode())
        time.sleep(0.1)
        response = []
        while True:
            line = self.serial.readline().decode('utf-8').strip()
            if line:
                response.append(line)
                if 'ok' in line or 'error' in line:
                    break
        return response

    def wakeup(self):
        self.serial.write(b'\x18')
        time.sleep(0.5)
        self.send_command("$X")

    def move_to(self, x, feed_rate=500):
        command = f"G1 X{x:.3f} F{feed_rate}"
        print(f"🔄 Перемещение в X={x} мм")
        return self.send_command(command)

    def move_relative(self, distance, feed_rate=500):
        self.send_command("G91")
        command = f"G1 X{distance:.3f} F{feed_rate}"
        print(f"🔄 Перемещение на {distance} мм")
        response = self.send_command(command)
        self.send_command("G90")
        return response

    def home(self):
        print("🏠 Поиск нуля...")
        self.send_command("$H")
        while True:
            line = self.serial.readline().decode('utf-8').strip()
            if 'ok' in line or 'error' in line:
                break
        print("✅ Ноль найден")

    def get_status(self):
        self.serial.write(b"?")
        time.sleep(0.1)
        status = self.serial.readline().decode('utf-8').strip()
        return status

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("🔌 Отключено")


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

if __name__ == "__main__":
    PORT = '/dev/cu.usbmodemFX2348N1'

    grbl = None

    try:
        grbl = GRBLController(port=PORT)

        print("\n🚀 Начинаем движение (0.5 см вправо/влево, медленная скорость)...")

        SPEED = 500  # мм/мин (в 2 раза медленнее)
        DISTANCE = 5  # мм (0.5 см = 5 мм) ← ИЗМЕНЕНО

        for i in range(5):  # 5 циклов
            print(f"\n--- Цикл {i + 1}/5 ---")

            # Движение вправо на 0.5 см (5 мм)
            grbl.move_to(DISTANCE, feed_rate=SPEED)
            time.sleep(1)

            # Движение влево на 0 мм (возврат, тоже 0.5 см)
            grbl.move_to(0, feed_rate=SPEED)
            time.sleep(1)

        print("\n✅ Готово!")

    except KeyboardInterrupt:
        print("\n⏹️ Остановка по Ctrl+C")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if grbl is not None:
            grbl.close()