import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from scipy.spatial.distance import cosine
import idx2numpy
import logging
import os
import pytesseract

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка EMNIST
def load_emnist(images_path, labels_path, mapping_path):
    """Загружает данные EMNIST из файлов IDX."""
    logging.info("Загрузка EMNIST...")
    try:
        # Проверяем, существуют ли файлы
        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Файл {images_path} не найден.")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Файл {labels_path} не найден.")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Файл {mapping_path} не найден.")

        # Загружаем изображения и метки
        images = idx2numpy.convert_from_file(images_path)
        labels = idx2numpy.convert_from_file(labels_path)

        # Загружаем маппинг (соответствие меток и символов)
        with open(mapping_path, 'r') as f:
            mapping = f.readlines()
        mapping = {int(line.split()[0]): line.split()[1] for line in mapping}

        logging.info("EMNIST успешно загружен.")
        return labels, images, mapping
    except Exception as e:
        logging.error(f"Ошибка при загрузке EMNIST: {e}")
        raise


# Предварительная обработка изображения
def preprocess_image(image):
    """Преобразует изображение в черно-белый формат и применяет фильтры."""
    logging.info("Предварительная обработка изображения...")
    try:
        # Конвертируем в черно-белый
        image = image.convert('L')
        # Применяем бинаризацию
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
        # Удаляем шум с помощью медианного фильтра
        image = image.filter(ImageFilter.MedianFilter(size=3))
        logging.info("Изображение успешно обработано.")
        return image
    except Exception as e:
        logging.error(f"Ошибка при обработке изображения: {e}")
        raise


# Разделение изображения на символы
def split_image_into_characters(image):
    """Разделяет изображение на отдельные символы."""
    logging.info("Разделение изображения на символы...")
    try:
        width, height = image.size
        characters = []
        start = 0
        min_char_width = 5  # Минимальная ширина символа (в пикселях)

        for x in range(width):
            column = [image.getpixel((x, y)) for y in range(height)]
            if all(pixel == 255 for pixel in column):  # Если столбец полностью белый
                if x > start:
                    # Проверяем, что ширина символа больше минимальной
                    if (x - start) >= min_char_width:
                        character = image.crop((start, 0, x, height))
                        characters.append(character)
                        logging.info(f"Найден символ: start={start}, end={x}, width={x - start}")
                    start = x + 1
                else:
                    start = x + 1

        # Обрабатываем последний символ
        if start < width and (width - start) >= min_char_width:
            character = image.crop((start, 0, width, height))
            characters.append(character)
            logging.info(f"Найден символ: start={start}, end={width}, width={width - start}")

        logging.info(f"Найдено {len(characters)} символов.")
        return characters
    except Exception as e:
        logging.error(f"Ошибка при разделении изображения: {e}")
        raise


# Преобразование изображения в вектор
def image_to_vector(image):
    """Преобразует изображение в вектор и нормализует его."""
    try:
        image = image.resize((28, 28))  # Ресайз до 28x28
        vector = np.array(image).flatten()  # Преобразуем в 1D вектор
        if vector.ndim != 1:
            raise ValueError(f"Ожидался одномерный вектор, но получен массив формы {vector.shape}")
        return vector / 255.0  # Нормализация
    except Exception as e:
        logging.error(f"Ошибка при преобразовании изображения: {e}")
        raise


# Сравнение изображений с использованием косинусного расстояния
def find_closest_emnist_image(input_vector, emnist_images, emnist_labels):
    """Находит ближайший символ в EMNIST."""
    logging.info("Поиск ближайшего символа в EMNIST...")
    try:
        min_distance = float('inf')
        best_label = None
        for i, emnist_image in enumerate(emnist_images):
            distance = cosine(input_vector, emnist_image)  # Косинусное расстояние
            if distance < min_distance:
                min_distance = distance
                best_label = emnist_labels[i]
        logging.info(f"Найден ближайший символ: {best_label}")
        return best_label
    except Exception as e:
        logging.error(f"Ошибка при поиске символа: {e}")
        raise


# Преобразование метки в символ
def label_to_char(label, mapping):
    """Преобразует числовую метку в символ с использованием маппинга."""
    return mapping.get(label, '?')  # Возвращает символ или '?', если метка неизвестна


# Распознавание текста на изображении с использованием EMNIST
def recognize_text_with_emnist(image_path, labels, images, mapping):
    """Распознает текст на изображении с использованием EMNIST."""
    logging.info(f"Начало распознавания текста для изображения: {image_path}")
    try:
        # Открываем входное изображение
        input_image = Image.open(image_path)

        # Предварительная обработка изображения
        processed_image = preprocess_image(input_image)

        # Разделяем изображение на символы
        characters = split_image_into_characters(processed_image)

        # Распознаем каждый символ
        recognized_text = ''
        for character in characters:
            input_vector = image_to_vector(character)
            recognized_label = find_closest_emnist_image(input_vector, images, labels)
            recognized_text += label_to_char(recognized_label, mapping)

        logging.info(f"Распознанный текст (EMNIST): {recognized_text}")
        return recognized_text
    except Exception as e:
        logging.error(f"Ошибка при распознавании текста (EMNIST): {e}")
        return None


# Распознавание текста на изображении с использованием Tesseract
def recognize_text_with_tesseract(image_path):
    """Распознает текст на изображении с использованием Tesseract."""
    logging.info(f"Начало распознавания текста для изображения: {image_path} (Tesseract)")
    try:
        # Открываем изображение
        image = Image.open(image_path)

        # Распознаем текст
        text = pytesseract.image_to_string(image, lang='rus')

        logging.info(f"Распознанный текст (Tesseract): {text}")
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка при распознавании текста (Tesseract): {e}")
        return None


if __name__ == "__main__":
    # Укажите пути к файлам EMNIST
    emnist_images_path = 'res/balanced_model/emnist-balanced-train-images-idx3-ubyte'
    emnist_labels_path = 'res/balanced_model/emnist-balanced-train-labels-idx1-ubyte'
    emnist_mapping_path = 'res/balanced_model/emnist-balanced-mapping.txt'

    # Укажите путь к папке с изображениями
    test_images_folder = 'res/img_for_tests'

    # Отладочная информация
    print("=== Отладочная информация ===")
    print(f"Путь к изображениям EMNIST: {emnist_images_path}")
    print(f"Путь к меткам EMNIST: {emnist_labels_path}")
    print(f"Путь к маппингу EMNIST: {emnist_mapping_path}")
    print(f"Путь к папке с тестовыми изображениями: {test_images_folder}")
    print(f"Папка с тестовыми изображениями существует: {os.path.exists(test_images_folder)}")
    print("============================")

    # Проверяем, существуют ли файлы EMNIST
    try:
        labels, images, mapping = load_emnist(emnist_images_path, emnist_labels_path, emnist_mapping_path)
    except Exception as e:
        print(f"Ошибка при загрузке EMNIST: {e}")
        exit(1)

    # Проверяем, существует ли папка с тестовыми изображениями
    if not os.path.exists(test_images_folder):
        print(f"Папка {test_images_folder} не найдена. Создайте папку и добавьте в неё изображения.")
        exit(1)

    # Обрабатываем все изображения в папке img_for_tests
    for filename in os.listdir(test_images_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(test_images_folder, filename)
            try:
                # Пытаемся распознать текст с помощью EMNIST
                recognized_text = recognize_text_with_emnist(image_path, labels, images, mapping)

                # Если EMNIST не справился, используем Tesseract
                if recognized_text is None or len(recognized_text.strip()) == 0:
                    recognized_text = recognize_text_with_tesseract(image_path)

                print(f"Изображение: {filename}, Распознанный текст: {recognized_text}")
            except Exception as e:
                print(f"Ошибка при обработке изображения {filename}: {e}")