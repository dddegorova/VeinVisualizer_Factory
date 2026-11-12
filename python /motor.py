import serial
import time


class GRBLController:
    def __init__(self, port='/dev/cu.usbmodem12345', baud_rate=115200):
        """
        Инициализация подключения к GRBL
        На Mac порт обычно называется /dev/cu.usbmodem... или /dev/cu.wchusbserial...
        """
        try:
            self.serial = serial.Serial(port, baud_rate, timeout=1)
            time.sleep(2)  # Ждем перезагрузки Arduino
            self.wakeup()
            print(f"✅ Подключено к {port}")
        except serial.SerialException as e:
            print(f" Ошибка подключения: {e}")
            print("💡 Проверьте, правильно ли указан порт (см. в Arduino IDE Tools → Port)")
            raise

    def send_command(self, command):
        """Отправка команды GRBL"""
        self.serial.write(f"{command}\n".encode())
        time.sleep(0.1)

        # Чтение ответа
        response = []
        while True:
            line = self.serial.readline().decode('utf-8').strip()
            if line:
                response.append(line)
                if 'ok' in line or 'error' in line:
                    break
        return response

    def wakeup(self):
        """Пробуждение GRBL"""
        self.serial.write(b'\x18')  # Команда сброса
        time.sleep(0.5)
        self.send_command("$X")  # Разблокировка

    def move_to(self, x, feed_rate=500):
        """
        Перемещение в абсолютную позицию X
        feed_rate - скорость перемещения (мм/мин)
        """
        command = f"G1 X{x:.3f} F{feed_rate}"
        print(f"🔄 Перемещение в X={x} мм")
        return self.send_command(command)

    def move_relative(self, distance, feed_rate=500):
        """
        Относительное перемещение на расстояние
        положительное - вправо, отрицательное - влево
        """
        self.send_command("G91")  # Относительный режим
        command = f"G1 X{distance:.3f} F{feed_rate}"
        print(f"🔄 Перемещение на {distance} мм")
        response = self.send_command(command)
        self.send_command("G90")  # Возврат в абсолютный режим
        return response

    def home(self):
        """Поиск домашнего положения (нуля)"""
        print(" Поиск нуля...")
        self.send_command("$H")
        # Ждем завершения homing
        while True:
            line = self.serial.readline().decode('utf-8').strip()
            if 'ok' in line or 'error' in line:
                break
        print("✅ Ноль найден")

    def get_status(self):
        """Получение текущего статуса"""
        self.serial.write(b"?")
        time.sleep(0.1)
        status = self.serial.readline().decode('utf-8').strip()
        return status

    def close(self):
        """Закрытие соединения"""
        self.serial.close()
        print(" Отключено")


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

if __name__ == "__main__":
    # ВАЖНО: Укажите ваш порт (посмотрите в Arduino IDE: Tools → Port)
    PORT = '/dev/cu.usbmodemFX2348N1'  # или /dev/cu.wchusbserial14101 для CH340

    try:
        # Создание контроллера
        grbl = GRBLController(port=PORT)

        # Нахождение нуля (опционально)
        # grbl.home()

        # Пример 1: Движение влево-вправо
        print("\n🚀 Начинаем движение...")

        for i in range(5):  # 5 циклов
            # Движение вправо на 50 мм
            grbl.move_to(50, feed_rate=1000)
            time.sleep(1)

            # Движение влево на 0 мм
            grbl.move_to(0, feed_rate=1000)
            time.sleep(1)

        # Пример 2: Плавное движение туда-обратно
        print("\n🔄 Плавное движение...")
        for pos in range(0, 100, 10):  # От 0 до 100 мм с шагом 10
            grbl.move_to(pos, feed_rate=500)
            time.sleep(0.5)

        for pos in range(100, 0, -10):  # Обратно
            grbl.move_to(pos, feed_rate=500)
            time.sleep(0.5)

        print("✅ Готово!")

    except KeyboardInterrupt:
        print("\n⏹️ Остановка по Ctrl+C")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        grbl.close()