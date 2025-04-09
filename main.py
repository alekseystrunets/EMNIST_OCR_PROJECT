import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import idx2numpy
import os

# Конфигурация
DATA_DIR = "res/balanced_model"
IMAGE_SIZE = 140
FONT_SIZE = 80


def generate_char_image(char, font_path="arial.ttf"):
    """Генерация четкого тестового изображения"""
    img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 255)
    draw = ImageDraw.Draw(img)

    try:
        # Пробуем разные шрифты для лучшего отображения
        for font in [font_path, "cour.ttf", "times.ttf"]:
            try:
                font = ImageFont.truetype(font, FONT_SIZE)
                break
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Точное центрирование с учетом метрик шрифта
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (IMAGE_SIZE - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (IMAGE_SIZE - (bbox[3] - bbox[1])) // 2 - bbox[1] - 5

    draw.text((x, y), char, fill=0, font=font)

    # Легкое размытие для реалистичности
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def enhanced_preprocessing(image):
    """Улучшенная предобработка изображения"""
    # Адаптивная бинаризация
    img = image.convert('L')
    threshold = np.array(img).mean() * 0.8
    img = img.point(lambda x: 0 if x < threshold else 255, '1')

    # Инверсия цветов (как в EMNIST)
    img = ImageOps.invert(img)

    # Удаление шумов
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # Умная обрезка с запасом
    bbox = img.getbbox()
    if bbox:
        margin = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 3
        img = img.crop((
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(img.width, bbox[2] + margin),
            min(img.height, bbox[3] + margin)
        ))

    # Пропорциональное масштабирование
    width, height = img.size
    scale = 20 / max(width, height)
    img = img.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    # Размещение на поле 28x28
    new_img = Image.new('L', (28, 28), 0)
    new_img.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    return new_img


def load_emnist():
    """Загрузка данных EMNIST с разделением цифр и букв"""
    images = idx2numpy.convert_from_file(os.path.join(DATA_DIR, 'emnist-balanced-train-images-idx3-ubyte'))
    labels = idx2numpy.convert_from_file(os.path.join(DATA_DIR, 'emnist-balanced-train-labels-idx1-ubyte'))

    with open(os.path.join(DATA_DIR, 'emnist-balanced-mapping.txt')) as f:
        mapping = {int(k): chr(int(v)) for line in f for k, v in [line.split()]}

    # Создаем отдельные индексы для цифр и букв
    digit_indices = np.where([mapping[l].isdigit() for l in labels])[0]
    letter_indices = np.where([mapping[l].islower() for l in labels])[0]

    return {
        'images': images,
        'labels': labels,
        'mapping': mapping,
        'digit_indices': digit_indices,
        'letter_indices': letter_indices
    }


def visualize_comparison(test_char, test_img, emnist_img, result_char):
    """Визуализация сравнения тестового изображения и результата"""
    # Создаем составное изображение
    comparison = Image.new('L', (IMAGE_SIZE * 2, IMAGE_SIZE), 255)
    comparison.paste(test_img, (0, 0))
    comparison.paste(emnist_img.resize((IMAGE_SIZE, IMAGE_SIZE)), (IMAGE_SIZE, 0))

    # Добавляем подписи
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), f"Тест: '{test_char}'", fill=0)
    draw.text((IMAGE_SIZE + 10, 10), f"EMNIST: '{result_char}'", fill=0)

    comparison.show(title="Сравнение")


def recognize_and_show(test_char, emnist_data):
    """Распознавание символа с визуализацией"""
    # Определяем тип символа
    is_digit = test_char.isdigit()
    search_indices = emnist_data['digit_indices'] if is_digit else emnist_data['letter_indices']

    # Генерация и сохранение тестового изображения
    test_img = generate_char_image(test_char)
    test_img.save(f"test_{test_char}.png")

    # Предобработка
    processed = enhanced_preprocessing(test_img)
    test_vector = np.array(processed).flatten() / 255.0

    # Поиск совпадений только в соответствующей категории
    emnist_vectors = emnist_data['images'][search_indices].reshape(len(search_indices), -1)
    test_norm = test_vector / np.linalg.norm(test_vector)
    emnist_norm = emnist_vectors / np.linalg.norm(emnist_vectors, axis=1)[:, None]

    # Косинусное расстояние
    similarities = np.dot(emnist_norm, test_norm)
    top_indices = np.argsort(-similarities)[:5]

    # Получаем результаты
    results = []
    for idx in top_indices:
        original_idx = search_indices[idx]
        char = emnist_data['mapping'][emnist_data['labels'][original_idx]]
        results.append((char, 1 - similarities[idx]))

    best_match = results[0][0]

    # Визуализация
    best_idx = search_indices[top_indices[0]]
    emnist_img = Image.fromarray((emnist_data['images'][best_idx] * 255).astype(np.uint8))
    visualize_comparison(test_char, test_img, emnist_img, best_match)

    return best_match, results


def main():
    print("Загрузка данных EMNIST...")
    emnist_data = load_emnist()

    # Тестируем по одному примеру цифры и буквы
    test_chars = ['5', 'b']  # Один цифра, одна буква

    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ РАСПОЗНАВАНИЯ")
    print("=" * 50)

    for char in test_chars:
        print(f"\nТестируем символ: '{char}'")
        result, top_matches = recognize_and_show(char, emnist_data)

        if result == char:
            print(f"✓ Верное распознавание: '{char}'")
        else:
            print(f"✗ Ошибка: ожидалось '{char}', получено '{result}'")

        print("\nТоп-5 совпадений:")
        for match, distance in top_matches:
            print(f"'{match}': {distance:.4f}")


if __name__ == "__main__":
    main()