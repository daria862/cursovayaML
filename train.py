import os
import torch
from ultralytics import YOLO

def main():
    # Конфигурация путей
    data_yaml_path = "skin_types_and_diseases/data.yaml"
    model_save_dir = "runs/detect/train"

    # 1. Проверка наличия и структуры датасета
    print("\n1. Проверка структуры датасета...")
    if not os.path.exists(data_yaml_path):
        print(f"Файл конфигурации {data_yaml_path} не найден!")
        return

    # Проверка существования папок с изображениями (опционально)
    with open(data_yaml_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
        train_path = config.get('train', '')
        val_path = config.get('val', '')

        for path in [train_path, val_path]:
            if path and os.path.exists(path):
                num_imgs = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
                print(f"   Найдено изображений в {os.path.basename(path)}: {num_imgs}")
            else:
                print(f"Путь не найден или пуст: {path}")

    print(f"   Классы для детектирования: {config.get('names', [])}")

    # Проверка доступного устройства
    print("\n2. Проверка вычислительного устройства...")
    device = 0 if torch.cuda.is_available() else 'cpu'
    if device == 0:
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU не обнаружен, обучение будет выполняться на CPU (может быть медленно)")