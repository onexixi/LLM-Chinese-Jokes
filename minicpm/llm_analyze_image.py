import os
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
import requests
import time
from sklearn.cluster import KMeans

# 读取配置文件
config = ConfigParser()
config.read('config.ini')

OLLAMA_API_URL = config.get('API', 'ollama_api_url')
OLLAMA_MODEL = config.get('API', 'ollama_modle')

video_folder = config.get('Paths', 'video_folder')
output_json = config.get('Paths', 'output_json')
cache_file = config.get('Paths', 'cache_file')
MAX_FRAMES = config.getint('Parameters', 'max_frames')
BATCH_SIZE = config.getint('Parameters', 'batch_size')


def extract_keyframes(video_path, max_frames):
    """智能提取视频的关键帧，使用K-means聚类"""
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_indices = []

    # 每秒采样一帧
    for i in range(0, frame_count, fps):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break

        # 将图像转换为特征向量
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32)).flatten()
        frames.append(resized)
        frame_indices.append(i)

    video.release()

    if len(frames) == 0:
        return []

    # 使用K-means聚类
    n_clusters = min(max_frames, len(frames))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(frames)

    # 找到每个簇的中心点
    cluster_centers = kmeans.cluster_centers_

    # 为每个簇选择最接近中心的帧
    keyframes = []
    for i in range(n_clusters):
        cluster_frames = [j for j, label in enumerate(kmeans.labels_) if label == i]
        if cluster_frames:
            center = cluster_centers[i]
            closest_frame_index = min(cluster_frames, key=lambda x: np.linalg.norm(frames[x] - center))
            keyframes.append(frame_indices[closest_frame_index])

    # 按时间顺序排序关键帧
    keyframes.sort()

    # 读取关键帧的实际图像数据
    video = cv2.VideoCapture(video_path)
    keyframe_images = []
    for frame_index in keyframes:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video.read()
        if ret:
            keyframe_images.append((frame_index, frame))
    video.release()

    return keyframe_images


def analyze_images(images):
    """批量使用Ollama API分析图像内容"""
    base64_images = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8') for img in images]

    analyses = []
    for base64_image in base64_images:
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": "请描述这个图片",
                    "stream": False,
                    "images": [base64_image]
                }
            )
            response.raise_for_status()
            result = response.json()
            print(result)
            analyses.append(result['response'])
        except Exception as e:
            print(f"Error in API call: {e}")
            analyses.append("Error in analysis")

    return analyses


def process_video(video_file):
    """处理单个视频文件"""
    video_path = os.path.join(video_folder, video_file)

    start_time = time.time()
    keyframes = extract_keyframes(video_path, MAX_FRAMES)
    extraction_time = time.time() - start_time

    video_results = []
    for i in range(0, len(keyframes), BATCH_SIZE):
        batch = keyframes[i:i + BATCH_SIZE]
        frame_counts, frames = zip(*batch)
        analyses = analyze_images(frames)

        for frame_count, analysis in zip(frame_counts, analyses):
            video_results.append({
                'frame': frame_count,
                'analysis': analysis
            })

    total_time = time.time() - start_time

    return video_file, video_results, extraction_time, total_time


def load_cache():
    """加载缓存的结果"""
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(results):
    """保存结果到缓存"""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def process_videos():
    """并行处理文件夹中的所有视频"""
    results = load_cache()

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_video = {executor.submit(process_video, video_file): video_file
                           for video_file in os.listdir(video_folder)
                           if video_file.endswith('.mp4') and video_file not in results}

        for future in tqdm(as_completed(future_to_video), total=len(future_to_video)):
            video_file = future_to_video[future]
            try:
                video_file, video_results, extraction_time, total_time = future.result()
                results[video_file] = {
                    'frames': video_results,
                    'extraction_time': extraction_time,
                    'total_processing_time': total_time
                }
                save_cache(results)  # 每处理完一个视频就保存一次缓存
            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    # 保存最终结果到JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    process_videos()
    print(f"Analysis complete. Results saved to {output_json}")