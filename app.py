import os
import uuid
import numpy as np
import cv2
import mediapipe as mp
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import subprocess
import shutil
from pathlib import Path
import time
import asyncio
from pydantic import BaseModel
import uvicorn

# Створюємо необхідні директорії
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("temp", exist_ok=True)

app = FastAPI(title="AI-система обробки відео з накладанням одягу")

# Налаштування шаблонів та статичних файлів
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Створюємо HTML шаблон
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>AI-система обробки відео з накладанням одягу</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        label {
            font-weight: bold;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .preview {
            margin-top: 20px;
            display: none;
        }
        .preview img, .preview video {
            max-width: 100%;
            border-radius: 5px;
            margin-top: 10px;
        }
        #progressBar {
            width: 100%;
            background-color: #ddd;
            height: 30px;
            border-radius: 4px;
            margin-top: 20px;
            display: none;
        }
        #progressBarFill {
            height: 100%;
            background-color: #4CAF50;
            border-radius: 4px;
            width: 0%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            transition: width 0.5s;
        }
        #result {
            margin-top: 20px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI-система обробки відео з накладанням одягу</h1>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="clothingFile">Завантажте фото одягу (JPG або PNG):</label>
                <input type="file" id="clothingFile" name="clothingFile" accept=".jpg,.jpeg,.png" required
                       onchange="previewImage('clothingFile', 'clothingPreview')">
                <div id="clothingPreview" class="preview">
                    <h3>Перегляд одягу:</h3>
                    <img id="clothingPreviewImg" src="#" alt="Перегляд одягу">
                </div>
            </div>
            
            <div class="form-group">
                <label for="videoFile">Завантажте відео користувача (MP4):</label>
                <input type="file" id="videoFile" name="videoFile" accept=".mp4" required
                       onchange="previewVideo('videoFile', 'videoPreview')">
                <div id="videoPreview" class="preview">
                    <h3>Перегляд відео:</h3>
                    <video id="videoPreviewVideo" controls></video>
                </div>
            </div>
            
            <button type="submit" class="btn">Обробити</button>
        </form>
        
        <div id="progressBar">
            <div id="progressBarFill">0%</div>
        </div>
        
        <div id="result">
            <h2>Результат обробки:</h2>
            <video id="resultVideo" controls width="100%"></video>
            <a id="downloadLink" class="btn" style="display:inline-block; margin-top: 10px; text-decoration: none; text-align: center;">
                Завантажити результат
            </a>
        </div>
    </div>

    <script>
        function previewImage(inputId, previewId) {
            const file = document.getElementById(inputId).files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById(previewId).style.display = 'block';
                    document.getElementById(previewId + 'Img').src = e.target.result;
                }
                reader.readAsDataURL(file);
            }
        }
        
        function previewVideo(inputId, previewId) {
            const file = document.getElementById(inputId).files[0];
            if (file) {
                const url = URL.createObjectURL(file);
                document.getElementById(previewId).style.display = 'block';
                const video = document.getElementById(previewId + 'Video');
                video.src = url;
                video.load();
            }
        }
        
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            formData.append('clothingFile', document.getElementById('clothingFile').files[0]);
            formData.append('videoFile', document.getElementById('videoFile').files[0]);
            
            // Show progress bar
            const progressBar = document.getElementById('progressBar');
            const progressBarFill = document.getElementById('progressBarFill');
            progressBar.style.display = 'block';
            
            // Simulate progress updates
            let progress = 0;
            const progressInterval = setInterval(() => {
                if (progress < 95) {
                    progress += Math.random() * 5;
                    progressBarFill.style.width = `${progress}%`;
                    progressBarFill.textContent = `${Math.round(progress)}%`;
                }
            }, 500);
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                clearInterval(progressInterval);
                
                if (response.ok) {
                    const data = await response.json();
                    progressBarFill.style.width = '100%';
                    progressBarFill.textContent = '100%';
                    
                    // Show result
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('resultVideo').src = data.video_url;
                    document.getElementById('downloadLink').href = data.video_url;
                    document.getElementById('downloadLink').download = 'processed_video.mp4';
                } else {
                    alert('Виникла помилка під час обробки відео');
                }
            } catch (error) {
                clearInterval(progressInterval);
                console.error('Error:', error);
                alert('Виникла помилка під час обробки відео');
            }
        });
    </script>
</body>
</html>
    """)

# Класи та функції для обробки відео

class ProcessingResult(BaseModel):
    video_url: str
    status: str

# MediaPipe для визначення частин тіла
class BodyPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
    
    def detect_pose(self, frame):
        # Конвертуємо BGR у RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Отримуємо результати
        results = self.pose.process(rgb_frame)
        
        return results
    
    def get_pose_landmarks(self, frame):
        results = self.detect_pose(frame)
        
        if results.pose_landmarks:
            landmarks = []
            h, w, _ = frame.shape
            for landmark in results.pose_landmarks.landmark:
                # Конвертуємо нормалізовані координати в пікселі
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
            
            return landmarks, results.segmentation_mask
        
        return None, None
    
    def close(self):
        self.pose.close()


# Симуляція класу для конвертації одягу у 3D модель
class Clothing3DConverter:
    def __init__(self):
        print("Ініціалізація моделі конвертації одягу у 3D")
        # В реальному проекті тут буде завантаження моделі
        
    def convert_to_3d(self, clothing_image, body_landmarks):
        print("Конвертація одягу у 3D модель...")
        
        # Це симуляція роботи моделі
        # В реальному проекті тут буде виклик до нейромережі (наприклад, PIFuHD)
        
        # Для демонстрації просто створимо маску з зображення одягу
        clothing_img = np.array(Image.open(clothing_image).convert('RGB'))
        h, w, _ = clothing_img.shape
        
        # Створюємо маску для одягу (спрощений підхід для демонстрації)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Визначаємо регіон де ймовірно є одяг (спрощена сегментація)
        hsv = cv2.cvtColor(clothing_img, cv2.COLOR_RGB2HSV)
        # Визначаємо маску на основі кольорів (це спрощений підхід)
        lower_bound = np.array([0, 20, 20])
        upper_bound = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Повертаємо оригінальне зображення та маску як прототип 3D моделі
        return {
            'original': clothing_img,
            'mask': color_mask,
            'texture': clothing_img,
            # В реальній системі тут будуть 3D дані (вершини, текстури тощо)
        }


# Клас для накладання одягу на відео
class ClothingOverlayer:
    def __init__(self):
        print("Ініціалізація системи накладання одягу")
    
    def overlay_on_frame(self, frame, clothing_3d_model, landmarks, segmentation_mask):
        print("Накладання одягу на кадр...")
        
        # Це спрощена версія для демонстрації
        # В реальному проекті тут буде виклик до рендера 3D моделі
        
        h, w, _ = frame.shape
        
        # Отримуємо маску одягу та адаптуємо її до розміру кадру
        clothing_mask = clothing_3d_model['mask']
        clothing_texture = clothing_3d_model['texture']
        
        # Змінюємо розмір одягу відповідно до тіла
        # У реальному проекті це буде складніша трансформація на основі пози
        if landmarks and len(landmarks) >= 33:  # MediaPipe Pose має 33 ключові точки
            # Визначаємо прямокутник торсу (спрощено)
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # Знаходимо границі торсу
            min_x = min(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
            max_x = max(left_shoulder[0], right_shoulder[0], left_hip[0], right_hip[0])
            min_y = min(left_shoulder[1], right_shoulder[1])
            max_y = max(left_hip[1], right_hip[1])
            
            # Переконуємося, що координати в межах кадру
            min_x = max(0, min_x - 20)
            max_x = min(w, max_x + 20)
            min_y = max(0, min_y - 20)
            max_y = min(h, max_y + 20)
            
            torso_width = max_x - min_x
            torso_height = max_y - min_y
            
            if torso_width > 0 and torso_height > 0:
                # Змінюємо розмір зображення одягу та маски
                resized_clothing = cv2.resize(clothing_texture, (torso_width, torso_height))
                resized_mask = cv2.resize(clothing_mask, (torso_width, torso_height))
                
                # Створюємо ROI для накладання
                roi = frame[min_y:max_y, min_x:max_x]
                
                # Накладаємо одяг на основі маски
                for i in range(torso_height):
                    for j in range(torso_width):
                        if i < roi.shape[0] and j < roi.shape[1] and i < resized_mask.shape[0] and j < resized_mask.shape[1]:
                            if resized_mask[i, j] > 50:  # Поріг для маски
                                roi[i, j] = resized_clothing[i, j]
                
                # Оновлюємо кадр з накладеним одягом
                frame[min_y:max_y, min_x:max_x] = roi
        
        return frame


# Клас для генерації фінального відео
class VideoGenerator:
    def __init__(self):
        print("Ініціалізація генератора відео")
    
    def generate_video(self, processed_frames, output_path, fps=60):
        print(f"Генерація відео з {len(processed_frames)} кадрів...")
        
        if not processed_frames:
            raise ValueError("Немає кадрів для генерації відео")
        
        # Отримуємо розміри з першого кадру
        h, w, _ = processed_frames[0].shape
        
        # Створюємо відеозапис
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Якщо кадрів менше 3, дублюємо їх
        if len(processed_frames) < 3:
            processed_frames = processed_frames * (3 // len(processed_frames) + 1)
        
        # Інтерполюємо кадри для отримання плавного відео
        total_frames = fps * 3  # 3 секунди відео при 60 fps
        
        # Генеруємо плавні переходи між ключовими кадрами
        for i in range(total_frames):
            # Визначаємо, які два ключові кадри ми використовуємо для інтерполяції
            frame_idx = (i / total_frames) * (len(processed_frames) - 1)
            idx1 = int(frame_idx)
            idx2 = min(idx1 + 1, len(processed_frames) - 1)
            alpha = frame_idx - idx1  # Коефіцієнт змішування
            
            # Лінійна інтерполяція між кадрами
            blended_frame = cv2.addWeighted(
                processed_frames[idx1], 1 - alpha,
                processed_frames[idx2], alpha, 0)
            
            out.write(blended_frame)
        
        out.release()
        print(f"Відео успішно згенеровано: {output_path}")
        return output_path


# Основний процесор відео для оркестрації всього процесу
class VideoProcessor:
    def __init__(self):
        self.body_detector = BodyPoseDetector()
        self.clothing_converter = Clothing3DConverter()
        self.clothing_overlayer = ClothingOverlayer()
        self.video_generator = VideoGenerator()
    
    async def process_video(self, clothing_file_path, video_file_path):
        print("Початок обробки відео...")
        
        # Створюємо унікальний ID для цієї обробки
        process_id = str(uuid.uuid4())
        output_dir = os.path.join("results", process_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Завантажуємо відео і отримуємо ключові кадри
        video = cv2.VideoCapture(video_file_path)
        
        if not video.isOpened():
            raise ValueError("Не вдалося відкрити відео")
        
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Обираємо 3 ключові кадри (початок, середина, кінець)
        key_frame_indices = [0, total_frames // 2, total_frames - 1]
        key_frames = []
        
        for idx in key_frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = video.read()
            if ret:
                key_frames.append(frame)
        
        video.release()
        
        if len(key_frames) < 1:
            raise ValueError("Не вдалося отримати ключові кадри з відео")
        
        print(f"Отримано {len(key_frames)} ключових кадрів")
        
        # Визначаємо пози на ключових кадрах
        landmarks_list = []
        segmentation_masks = []
        
        for frame in key_frames:
            landmarks, segmentation_mask = self.body_detector.get_pose_landmarks(frame)
            if landmarks:
                landmarks_list.append(landmarks)
                segmentation_masks.append(segmentation_mask)
            else:
                # Якщо не виявлено пози, використовуємо заглушки
                landmarks_list.append(None)
                segmentation_masks.append(None)
        
        # Конвертуємо одяг у 3D модель
        clothing_3d_model = self.clothing_converter.convert_to_3d(
            clothing_file_path, landmarks_list[0] if landmarks_list else None
        )
        
        # Накладаємо одяг на ключові кадри
        processed_frames = []
        
        for i, frame in enumerate(key_frames):
            if i < len(landmarks_list) and landmarks_list[i]:
                processed_frame = self.clothing_overlayer.overlay_on_frame(
                    frame.copy(), clothing_3d_model, 
                    landmarks_list[i], segmentation_masks[i] if i < len(segmentation_masks) else None
                )
                processed_frames.append(processed_frame)
            else:
                processed_frames.append(frame.copy())
        
        # Генеруємо фінальне відео
        output_path = os.path.join(output_dir, "processed_video.mp4")
        self.video_generator.generate_video(processed_frames, output_path, fps=60)
        
        # Закриваємо детектор пози
        self.body_detector.close()
        
        return output_path


# Маршрути FastAPI
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/process", response_model=ProcessingResult)
async def process_video(
    clothingFile: UploadFile = File(...),
    videoFile: UploadFile = File(...)
):
    if not clothingFile.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Тільки файли JPG або PNG дозволені для одягу")
    
    if not videoFile.filename.lower().endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Тільки файли MP4 дозволені для відео")
    
    # Створюємо тимчасові шляхи для завантажених файлів
    clothing_path = os.path.join("uploads", f"{uuid.uuid4()}_{clothingFile.filename}")
    video_path = os.path.join("uploads", f"{uuid.uuid4()}_{videoFile.filename}")
    
    # Зберігаємо завантажені файли
    with open(clothing_path, "wb") as f:
        f.write(await clothingFile.read())
    
    with open(video_path, "wb") as f:
        f.write(await videoFile.read())
    
    try:
        # Обробляємо відео
        processor = VideoProcessor()
        output_video_path = await processor.process_video(clothing_path, video_path)
        
        # Формуємо URL для доступу до відео
        video_url = f"/results/{os.path.basename(os.path.dirname(output_video_path))}/processed_video.mp4"
        
        return ProcessingResult(
            video_url=video_url,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")
    
    finally:
        # Видаляємо тимчасові файли
        if os.path.exists(clothing_path):
            os.remove(clothing_path)
        if os.path.exists(video_path):
            os.remove(video_path)

# Маршрут для доступу до результатів
@app.get("/results/{process_id}/{filename}")
async def get_result(process_id: str, filename: str):
    file_path = os.path.join("results", process_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Файл не знайдено")

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 