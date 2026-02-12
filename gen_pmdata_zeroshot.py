#!/usr/bin/env python3
"""
PMData 데이터셋 생성 (논문 Zero-shot 형식)
유저 정보 포함 + Bullet point 형식
"""

import json
import os
import csv
import re
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration
MODE = "train"
DATA = "PMData"
DATA_PATH = "/home/khlee/fine-tuning/Health-LLM-datas/data/pmdata"
PARTICIPANT_OVERVIEW = "/home/khlee/fine-tuning/Health-LLM-datas/data/pmdata/participant-overview.xlsx"
OUTPUT_DIR = "/home/khlee/fine-tuning/Health-LLM-datas/output"

def json_reader(file_name):
    with open(file_name) as f:
        return json.load(f)

def csv_reader(file_name):
    return csv.reader(open(file_name, 'r'))

# Load participant demographics
print("=== Loading Participant Demographics ===")
df = pd.read_excel(PARTICIPANT_OVERVIEW)
participant_info = {}

for idx in range(1, len(df)):  # Skip header row (row 0)
    pid = df.iloc[idx, 0]  # Participant ID (p01, p02, ...)
    age = df.iloc[idx, 1]  # Age
    height = df.iloc[idx, 2]  # Height (cm)
    gender = df.iloc[idx, 3]  # Gender (male/female)

    participant_info[pid] = {
        'age': int(age),
        'height': int(height),
        'gender': str(gender).strip()
    }

print(f"✓ Loaded {len(participant_info)} participants")
for pid, info in list(participant_info.items())[:3]:
    print(f"  {pid}: {info['age']}y {info['gender']} {info['height']}cm")

# Process each subtask
SUBTASKS = ["stress", "readiness", "sleep_quality", "fatigue"]

for SUBTASK in SUBTASKS:
    print(f"\n{'=' * 80}")
    print(f"Processing: PMData_{SUBTASK}")
    print('=' * 80)

    final_data = []

    for dir1 in tqdm(os.listdir(DATA_PATH), desc=SUBTASK):
        if "." in dir1 or dir1 not in participant_info:
            continue

        # Get participant demographics
        demographics = participant_info[dir1]
        age = demographics['age']
        height = demographics['height']
        gender = demographics['gender']

        fpath1 = os.path.join(DATA_PATH, dir1)

        for dir2 in os.listdir(fpath1):
            fpath2 = os.path.join(fpath1, dir2)

            if dir2 == 'fitbit':
                # Read fitbit data
                try:
                    heart_rate_data = json_reader(os.path.join(fpath2, 'resting_heart_rate.json'))
                except:
                    continue

                exercise_data = json_reader(os.path.join(fpath2, 'exercise.json'))
                sleep_data = json_reader(os.path.join(fpath2, 'sleep.json'))

            elif dir2 == 'pmsys':
                # Read wellness data
                wellness_data = csv_reader(os.path.join(fpath2, "wellness.csv"))
                wellness_dict = {
                    'effective_time_frame': [], 'fatigue': [], 'mood': [],
                    'readiness': [], 'sleep_duration_h': [], 'sleep_quality': [],
                    'stress': []
                }

                for i, data in enumerate(wellness_data):
                    if i == 0:
                        continue

                    date = data[0][:10] + "_" + data[0][11:][:-1].split(".")[0]
                    wellness_dict['effective_time_frame'].append(date)
                    wellness_dict['fatigue'].append(data[1])
                    wellness_dict['mood'].append(data[2])
                    wellness_dict['readiness'].append(data[3])
                    wellness_dict['sleep_duration_h'].append(data[4])
                    wellness_dict['sleep_quality'].append(data[5])
                    wellness_dict['stress'].append(data[-1])

        # Process each wellness record
        for d, f, m, r, sd, sq, s in zip(
            wellness_dict['effective_time_frame'],
            wellness_dict['fatigue'],
            wellness_dict['mood'],
            wellness_dict['readiness'],
            wellness_dict['sleep_duration_h'],
            wellness_dict['sleep_quality'],
            wellness_dict['stress']
        ):
            new_d = datetime.strptime(d, '%Y-%m-%d_%H:%M:%S')

            # Collect 2-weeks historical data
            exercise_hist = []
            for e_data in exercise_data:
                e_date = e_data['startTime'][:10] + "_" + e_data['startTime'][11:]
                new_ed = datetime.strptime(e_date, '%Y-%m-%d_%H:%M:%S')

                if (new_d > new_ed) and (new_d - new_ed) < timedelta(days=14):
                    try:
                        steps = float(e_data['steps'])
                        calories = float(e_data['calories'])
                        exercise_hist.append([new_ed, calories, steps])
                    except:
                        continue

            sleep_hist = []
            for s_data in sleep_data:
                s_date = s_data['startTime'][:10] + "_" + s_data['startTime'][11:]
                new_sd = datetime.strptime(s_date, '%Y-%m-%d_%H:%M:%S')

                if (new_d > new_sd) and (new_d - new_sd) < timedelta(days=14):
                    sleep_duration = float(s_data['duration']) / 1000 / 60  # minutes
                    sleep_hist.append([new_sd, sleep_duration])

            hr_hist = []
            for hr_data in heart_rate_data:
                hr_date = hr_data['dateTime'][:10] + "_" + hr_data['dateTime'][11:]
                new_hrd = datetime.strptime(hr_date, '%Y-%m-%d_%H:%M:%S')

                if (new_d > new_hrd) and (new_d - new_hrd) < timedelta(days=14):
                    rhr = float(hr_data['value']['value'])
                    hr_hist.append([new_hrd, rhr])

            # Skip if insufficient data
            if len(exercise_hist) == 0 or len(sleep_hist) == 0 or len(hr_hist) == 0:
                continue

            # Extract lists
            steps_list = [x[-1] for x in exercise_hist]
            calories_list = [x[-2] for x in exercise_hist]
            rhr_list = [x[-1] for x in hr_hist]
            sleep_list = [x[-1] for x in sleep_hist]

            # Set task-specific parameters
            if SUBTASK == "readiness":
                range1, range2 = 0, 10
                label = r
            elif SUBTASK == "stress":
                range1, range2 = 1, 5
                label = s
            elif SUBTASK == "sleep_quality":
                range1, range2 = 1, 5
                label = sq
            elif SUBTASK == "fatigue":
                range1, range2 = 1, 5
                label = f

            # Create zero-shot prompt (논문 형식)
            instruction = "You are an intelligent healthcare agent."

            question = f"The user is {age}-year-old {gender} with {height} cm. The analysis of recent 2-weeks sensor readings show:\n"
            question += f"• Steps: {steps_list} steps\n"
            question += f"• Burned Calories: {calories_list} calories\n"
            question += f"• Resting Heart Rate: {rhr_list} beats/min\n"
            question += f"• Sleep Minutes: {sleep_list} minutes\n"
            question += f"• Mood: {m} out of 5\n"
            question += f"\nIn this regard, can you predict user's {SUBTASK.replace('_', ' ')} between {range1} and {range2}?"

            # Output: just the number (논문 형식)
            answer = str(label)

            final_data.append({
                'instruction': instruction,
                'input': question,
                'output': answer,
                'participant_id': dir1  # For reference
            })

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"PMData_{SUBTASK}_zeroshot.json")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated {len(final_data)} samples")
    print(f"✓ Saved to: {output_file}")

print("\n" + "=" * 80)
print("ALL DONE!")
print("=" * 80)
