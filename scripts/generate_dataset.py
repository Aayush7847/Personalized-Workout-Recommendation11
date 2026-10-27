"""
Generates data/exercises.csv from a curated exercise catalogue.

The project already ships the generated CSV, so you never need to run this.
It is included for transparency (so examiners can see exactly how every
column was derived) and so the dataset can be rebuilt reproducibly:

    python scripts/generate_dataset.py

Each catalogue row is hand-written (name, muscle group, equipment,
difficulty, type, goals, mechanics, movement pattern). The remaining columns
(sets, reps, rest, duration, calories, instructions, safety notes, tags) are
derived with explicit, documented rules so the dataset is internally
consistent.

NOTE: calorie values are APPROXIMATE EDUCATIONAL ESTIMATES (MET-style
formula for a 70 kg reference person). They are not medically accurate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.config import GOAL_PRESCRIPTION, REFERENCE_BODY_WEIGHT_KG  # noqa: E402
from utils.timing import estimate_exercise_seconds  # noqa: E402

MUSCLE = {"CH": "Chest", "BK": "Back", "SH": "Shoulders", "BI": "Biceps",
          "TR": "Triceps", "LG": "Legs", "GL": "Glutes", "CO": "Core",
          "FB": "Full Body", "CA": "Cardio"}
EQUIP = {"NE": "No Equipment", "DB": "Dumbbells", "BB": "Barbell",
         "RB": "Resistance Band", "CB": "Cable", "MC": "Machine", "BN": "Bench",
         "KB": "Kettlebell", "TM": "Treadmill", "PB": "Pull-up Bar",
         "FG": "Full Gym"}
DIFF = {"B": "Beginner", "I": "Intermediate", "A": "Advanced"}
TYPE = {"S": "Strength", "C": "Cardio", "H": "HIIT", "M": "Mobility", "K": "Core"}
GOAL = {"WL": "Weight Loss", "MG": "Muscle Gain", "ST": "Strength",
        "GF": "General Fitness", "EN": "Endurance"}
MECH = {"C": "Compound", "I": "Isolation"}

# name | muscle | equipment | difficulty | type | goals (primary first) | mechanics | pattern
CATALOGUE = """
Push-Up|CH|NE|B|S|GF,MG,EN|C|h_push
Knee Push-Up|CH|NE|B|S|GF,EN|C|h_push
Incline Push-Up|CH|BN|B|S|GF,EN|C|h_push
Decline Push-Up|CH|BN|I|S|MG,GF|C|h_push
Wide-Grip Push-Up|CH|NE|B|S|GF,MG|C|h_push
Archer Push-Up|CH|NE|A|S|ST,MG|C|h_push
Plyometric Push-Up|CH|NE|A|H|ST,WL|C|plyo
Bench Press|CH|BB|I|S|MG,ST|C|h_push
Incline Barbell Bench Press|CH|BB|I|S|MG,ST|C|incline_push
Decline Barbell Bench Press|CH|BB|I|S|MG|C|h_push
Paused Barbell Bench Press|CH|BB|A|S|ST,MG|C|h_push
Dumbbell Bench Press|CH|DB|B|S|MG,GF|C|h_push
Incline Dumbbell Press|CH|DB|I|S|MG|C|incline_push
Dumbbell Floor Press|CH|DB|B|S|MG,ST|C|h_push
Dumbbell Fly|CH|DB|I|S|MG|I|fly
Incline Dumbbell Fly|CH|DB|I|S|MG|I|fly
Dumbbell Pullover|CH|DB|I|S|MG|I|fly
Dumbbell Squeeze Press|CH|DB|I|S|MG|C|h_push
Cable Fly|CH|CB|I|S|MG|I|fly
Low-to-High Cable Fly|CH|CB|I|S|MG|I|fly
High-to-Low Cable Fly|CH|CB|I|S|MG|I|fly
Machine Chest Press|CH|MC|B|S|MG,GF|C|h_push
Pec Deck Fly|CH|MC|B|S|MG|I|fly
Smith Machine Incline Press|CH|MC|I|S|MG,ST|C|incline_push
Chest Dips|CH|FG|I|S|MG,ST|C|dip
Weighted Chest Dips|CH|FG|A|S|ST,MG|C|dip
Resistance Band Chest Press|CH|RB|B|S|GF,MG|C|h_push
Resistance Band Chest Fly|CH|RB|B|S|GF,MG|I|fly
Kettlebell Floor Press|CH|KB|I|S|ST,MG|C|h_push
Doorway Chest Stretch|CH|NE|B|M|GF|I|mobility
Pull-Up|BK|PB|I|S|ST,MG|C|v_pull
Wide-Grip Pull-Up|BK|PB|A|S|MG,ST|C|v_pull
Weighted Pull-Up|BK|PB|A|S|ST|C|v_pull
Negative Pull-Up|BK|PB|B|S|ST,GF|C|v_pull
Dead Hang|BK|PB|B|S|GF,EN|I|hang
Table Inverted Row|BK|NE|B|S|GF,MG|C|h_pull
Superman Hold|BK|NE|B|S|GF,EN|I|back_ext
Prone Y-T-W Raise|BK|NE|B|S|GF|I|rear_delt
Barbell Bent-Over Row|BK|BB|I|S|MG,ST|C|h_pull
Pendlay Row|BK|BB|A|S|ST,MG|C|h_pull
Barbell Deadlift|BK|BB|I|S|ST,MG|C|hinge
Rack Pull|BK|BB|I|S|ST|C|hinge
Barbell Shrug|BK|BB|B|S|MG,ST|I|shrug
Dumbbell Shrug|BK|DB|B|S|MG|I|shrug
One-Arm Dumbbell Row|BK|DB|B|S|MG,GF|C|h_pull
Chest-Supported Dumbbell Row|BK|DB|I|S|MG|C|h_pull
Renegade Row|BK|DB|A|S|ST,GF|C|h_pull
Lat Pulldown|BK|CB|B|S|MG,GF|C|v_pull
Close-Grip Lat Pulldown|BK|CB|B|S|MG|C|v_pull
Seated Cable Row|BK|CB|B|S|MG,GF|C|h_pull
Straight-Arm Cable Pulldown|BK|CB|I|S|MG|I|v_pull
Machine Row|BK|MC|B|S|MG,GF|C|h_pull
T-Bar Row|BK|MC|I|S|MG,ST|C|h_pull
Assisted Pull-Up Machine|BK|MC|B|S|GF,MG|C|v_pull
Back Extension|BK|MC|B|S|GF,ST|I|back_ext
Resistance Band Row|BK|RB|B|S|GF,EN|C|h_pull
Resistance Band Lat Pulldown|BK|RB|B|S|GF|C|v_pull
Resistance Band Pull-Apart|BK|RB|B|S|GF|I|rear_delt
Kettlebell Gorilla Row|BK|KB|I|S|MG,ST|C|h_pull
Reverse Snow Angel|BK|NE|B|M|GF|I|mobility
Cat-Cow Stretch|BK|NE|B|M|GF|I|mobility
Thoracic Spine Rotation|BK|NE|B|M|GF|I|mobility
Child's Pose|BK|NE|B|M|GF|I|mobility
Barbell Overhead Press|SH|BB|I|S|ST,MG|C|v_push
Push Press|SH|BB|A|S|ST|C|v_push
Seated Dumbbell Shoulder Press|SH|DB|B|S|MG,GF|C|v_push
Arnold Press|SH|DB|I|S|MG|C|v_push
Dumbbell Z Press|SH|DB|A|S|ST,MG|C|v_push
Dumbbell Lateral Raise|SH|DB|B|S|MG|I|lateral_raise
Dumbbell Front Raise|SH|DB|B|S|MG,GF|I|lateral_raise
Bent-Over Rear Delt Fly|SH|DB|I|S|MG|I|rear_delt
Dumbbell Upright Row|SH|DB|I|S|MG|C|upright_row
Barbell Upright Row|SH|BB|I|S|MG|C|upright_row
Landmine Press|SH|BB|I|S|ST,GF|C|v_push
Cable Lateral Raise|SH|CB|I|S|MG|I|lateral_raise
Cable Front Raise|SH|CB|B|S|MG|I|lateral_raise
Face Pull|SH|CB|B|S|GF,MG|I|rear_delt
Machine Shoulder Press|SH|MC|B|S|MG,GF|C|v_push
Reverse Pec Deck|SH|MC|B|S|MG|I|rear_delt
Pike Push-Up|SH|NE|I|S|ST,GF|C|v_push
Handstand Push-Up|SH|NE|A|S|ST|C|v_push
Wall Walk|SH|NE|A|S|ST,GF|C|v_push
Resistance Band Overhead Press|SH|RB|B|S|GF|C|v_push
Resistance Band Lateral Raise|SH|RB|B|S|GF,MG|I|lateral_raise
Resistance Band Face Pull|SH|RB|B|S|GF|I|rear_delt
Kettlebell Overhead Press|SH|KB|I|S|ST,GF|C|v_push
Kettlebell Bottoms-Up Press|SH|KB|A|S|ST|C|v_push
Kettlebell Halo|SH|KB|B|M|GF|I|mobility
Arm Circles|SH|NE|B|M|GF|I|mobility
Wall Slides|SH|NE|B|M|GF|I|mobility
Band Shoulder Dislocates|SH|RB|B|M|GF|I|mobility
Barbell Curl|BI|BB|B|S|MG|I|curl
EZ-Bar Curl|BI|BB|B|S|MG|I|curl
Barbell Preacher Curl|BI|BB|I|S|MG|I|curl
Drag Curl|BI|BB|I|S|MG|I|curl
Reverse Barbell Curl|BI|BB|I|S|MG|I|curl
21s Barbell Curl|BI|BB|A|S|MG,EN|I|curl
Dumbbell Bicep Curl|BI|DB|B|S|MG,GF|I|curl
Hammer Curl|BI|DB|B|S|MG|I|curl
Incline Dumbbell Curl|BI|DB|I|S|MG|I|curl
Concentration Curl|BI|DB|B|S|MG|I|curl
Zottman Curl|BI|DB|I|S|MG|I|curl
Spider Curl|BI|DB|I|S|MG|I|curl
Cable Curl|BI|CB|B|S|MG|I|curl
Cable Rope Hammer Curl|BI|CB|B|S|MG|I|curl
Bayesian Cable Curl|BI|CB|I|S|MG|I|curl
Machine Preacher Curl|BI|MC|B|S|MG|I|curl
Resistance Band Curl|BI|RB|B|S|GF,MG|I|curl
Resistance Band Hammer Curl|BI|RB|B|S|GF|I|curl
Chin-Up|BI|PB|I|S|MG,ST|C|v_pull
Kettlebell Curl|BI|KB|B|S|MG|I|curl
Towel Isometric Curl|BI|NE|B|S|GF|I|curl
Table Underhand Row|BI|NE|I|S|GF,MG|C|h_pull
Biceps Wall Stretch|BI|NE|B|M|GF|I|mobility
Close-Grip Bench Press|TR|BB|I|S|ST,MG|C|h_push
Skull Crusher|TR|BB|I|S|MG|I|tri_ext
JM Press|TR|BB|A|S|ST,MG|C|tri_ext
Overhead Dumbbell Triceps Extension|TR|DB|B|S|MG|I|tri_ext
Dumbbell Kickback|TR|DB|B|S|MG,GF|I|tri_ext
Dumbbell Skull Crusher|TR|DB|I|S|MG|I|tri_ext
Tate Press|TR|DB|A|S|MG|I|tri_ext
Tricep Pushdown|TR|CB|B|S|MG|I|tri_ext
Rope Pushdown|TR|CB|B|S|MG|I|tri_ext
Overhead Cable Triceps Extension|TR|CB|I|S|MG|I|tri_ext
Single-Arm Reverse-Grip Pushdown|TR|CB|I|S|MG|I|tri_ext
Machine Dip|TR|MC|B|S|MG,GF|C|dip
Bench Dip|TR|BN|B|S|GF,EN|C|dip
Diamond Push-Up|TR|NE|I|S|MG,GF|C|h_push
Close-Grip Knee Push-Up|TR|NE|B|S|GF|C|h_push
Parallel Bar Triceps Dip|TR|FG|I|S|MG,ST|C|dip
Resistance Band Pushdown|TR|RB|B|S|GF|I|tri_ext
Resistance Band Overhead Extension|TR|RB|B|S|GF,MG|I|tri_ext
Kettlebell Overhead Extension|TR|KB|I|S|MG|I|tri_ext
Sphinx Push-Up|TR|NE|A|S|ST,MG|I|tri_ext
Triceps Overhead Stretch|TR|NE|B|M|GF|I|mobility
Bodyweight Squat|LG|NE|B|S|GF,EN|C|squat
Jump Squat|LG|NE|I|H|WL,EN|C|plyo
Pistol Squat|LG|NE|A|S|ST|C|squat
Sissy Squat|LG|NE|A|S|MG|I|squat
Wall Sit|LG|NE|B|S|EN,GF|I|iso_hold
Walking Lunge|LG|NE|B|S|GF,EN|C|lunge
Reverse Lunge|LG|NE|B|S|GF|C|lunge
Jumping Lunge|LG|NE|A|H|WL,EN|C|plyo
Standing Calf Raise|LG|NE|B|S|GF|I|calf
Nordic Hamstring Curl|LG|NE|A|S|ST|I|leg_iso
Bulgarian Split Squat|LG|BN|I|S|MG,ST|C|lunge
Step-Up|LG|BN|B|S|GF,EN|C|lunge
Box Jump|LG|BN|I|H|ST,WL|C|plyo
Barbell Back Squat|LG|BB|I|S|ST,MG|C|squat
Barbell Front Squat|LG|BB|A|S|ST,MG|C|squat
Pause Squat|LG|BB|A|S|ST|C|squat
Barbell Romanian Deadlift|LG|BB|I|S|MG,ST|C|hinge
Barbell Lunge|LG|BB|I|S|MG,ST|C|lunge
Goblet Squat|LG|DB|B|S|MG,GF|C|squat
Dumbbell Lunge|LG|DB|B|S|MG,GF|C|lunge
Dumbbell Romanian Deadlift|LG|DB|B|S|MG|C|hinge
Dumbbell Step-Up|LG|DB|I|S|MG,GF|C|lunge
Dumbbell Bulgarian Split Squat|LG|DB|I|S|MG|C|lunge
Leg Press|LG|MC|B|S|MG,ST|C|squat
Hack Squat|LG|MC|I|S|MG,ST|C|squat
Smith Machine Squat|LG|MC|B|S|MG,ST|C|squat
Leg Extension|LG|MC|B|S|MG|I|leg_iso
Lying Leg Curl|LG|MC|B|S|MG|I|leg_iso
Seated Leg Curl|LG|MC|B|S|MG|I|leg_iso
Seated Calf Raise|LG|MC|B|S|MG|I|calf
Standing Machine Calf Raise|LG|MC|B|S|MG|I|calf
Kettlebell Goblet Squat|LG|KB|B|S|GF,MG|C|squat
Double Kettlebell Front Squat|LG|KB|A|S|ST|C|squat
Resistance Band Squat|LG|RB|B|S|GF|C|squat
Banded Lying Leg Curl|LG|RB|B|S|GF|I|leg_iso
Cossack Squat|LG|NE|I|M|GF|C|mobility
Deep Squat Hold|LG|NE|B|M|GF|C|mobility
Standing Hamstring Stretch|LG|NE|B|M|GF|I|mobility
Ankle Mobility Drill|LG|NE|B|M|GF|I|mobility
Glute Bridge|GL|NE|B|S|GF,MG|I|glute_bridge
Single-Leg Glute Bridge|GL|NE|I|S|MG,GF|I|glute_bridge
Frog Pump|GL|NE|B|S|GF,EN|I|glute_bridge
Fire Hydrant|GL|NE|B|S|GF|I|abduction
Donkey Kick|GL|NE|B|S|GF|I|abduction
Curtsy Lunge|GL|NE|B|S|GF|C|lunge
Barbell Hip Thrust|GL|BB|I|S|MG,ST|C|glute_bridge
Barbell Good Morning|GL|BB|A|S|ST|C|hinge
Sumo Deadlift|GL|BB|A|S|ST,MG|C|hinge
Dumbbell Hip Thrust|GL|DB|B|S|MG|C|glute_bridge
Dumbbell Sumo Squat|GL|DB|B|S|MG,GF|C|squat
Dumbbell Curtsy Lunge|GL|DB|I|S|MG|C|lunge
B-Stance Romanian Deadlift|GL|DB|I|S|MG|C|hinge
Banded Glute Bridge|GL|RB|B|S|GF|I|glute_bridge
Banded Lateral Walk|GL|RB|B|S|GF,EN|I|abduction
Banded Clamshell|GL|RB|B|S|GF|I|abduction
Cable Glute Kickback|GL|CB|I|S|MG|I|abduction
Cable Pull-Through|GL|CB|I|S|MG,ST|C|hinge
Hip Abduction Machine|GL|MC|B|S|MG|I|abduction
Reverse Hyperextension|GL|MC|I|S|MG,ST|I|back_ext
Smith Machine Hip Thrust|GL|MC|I|S|MG|C|glute_bridge
Glute-Ham Raise|GL|FG|A|S|ST|I|leg_iso
Kettlebell Swing|GL|KB|I|H|WL,EN|C|hinge
Single-Leg Kettlebell Deadlift|GL|KB|I|S|GF,ST|C|hinge
Half-Kneeling Hip Flexor Stretch|GL|NE|B|M|GF|I|mobility
90/90 Hip Switch|GL|NE|I|M|GF|I|mobility
Pigeon Stretch|GL|NE|B|M|GF|I|mobility
Plank|CO|NE|B|K|GF,EN|I|plank
Side Plank|CO|NE|B|K|GF,EN|I|plank
Hollow Body Hold|CO|NE|I|K|ST,GF|I|plank
Shoulder Tap Plank|CO|NE|I|K|GF,EN|I|plank
L-Sit Hold|CO|NE|A|K|ST|I|iso_hold
Dead Bug|CO|NE|B|K|GF|I|stability
Bird Dog|CO|NE|B|K|GF|I|stability
Crunch|CO|NE|B|K|GF,MG|I|core_flex
Bicycle Crunch|CO|NE|B|K|WL,GF|I|rotation
Reverse Crunch|CO|NE|B|K|GF|I|core_flex
Lying Leg Raise|CO|NE|I|K|MG,GF|I|core_flex
V-Up|CO|NE|I|K|GF,ST|I|core_flex
Flutter Kicks|CO|NE|B|K|EN,WL|I|core_flex
Russian Twist|CO|NE|B|K|GF|I|rotation
Plank Jacks|CO|NE|I|H|WL,EN|C|plyo
Hanging Knee Raise|CO|PB|I|K|MG,ST|I|core_flex
Hanging Leg Raise|CO|PB|A|K|ST,MG|I|core_flex
Toes-to-Bar|CO|PB|A|K|ST|I|core_flex
Weighted Russian Twist|CO|DB|I|K|MG|I|rotation
Dumbbell Side Bend|CO|DB|B|K|MG|I|rotation
Suitcase Carry|CO|DB|I|K|ST,GF|C|carry
Barbell Rollout|CO|BB|A|K|ST|I|core_flex
Landmine Rotation|CO|BB|I|K|ST,GF|I|rotation
Decline Sit-Up|CO|BN|I|K|MG|I|core_flex
Cable Crunch|CO|CB|I|K|MG|I|core_flex
Cable Woodchop|CO|CB|I|K|GF,ST|I|rotation
Pallof Press|CO|CB|B|K|GF,ST|I|stability
Ab Crunch Machine|CO|MC|B|K|MG|I|core_flex
Captain's Chair Leg Raise|CO|MC|I|K|MG|I|core_flex
Resistance Band Pallof Press|CO|RB|B|K|GF|I|stability
Resistance Band Woodchop|CO|RB|B|K|GF|I|rotation
Kettlebell Windmill|CO|KB|A|K|ST,GF|I|rotation
Cobra Stretch|CO|NE|B|M|GF|I|mobility
Burpee|FB|NE|I|H|WL,EN|C|plyo
Half Burpee|FB|NE|B|H|WL,GF|C|plyo
Burpee Broad Jump|FB|NE|A|H|WL,EN|C|plyo
Squat Thrust|FB|NE|B|H|WL|C|plyo
Bear Crawl|FB|NE|I|H|GF,EN|C|crawl
Running|FB|NE|B|C|WL,EN|C|cardio_steady
Inchworm|FB|NE|B|M|GF|C|mobility
World's Greatest Stretch|FB|NE|B|M|GF|C|mobility
Sun Salutation Flow|FB|NE|B|M|GF,EN|C|mobility
Downward Dog to Cobra|FB|NE|B|M|GF|C|mobility
Barbell Thruster|FB|BB|A|H|WL,ST|C|thruster
Barbell Clean and Press|FB|BB|A|S|ST|C|olympic
Power Clean|FB|BB|A|S|ST|C|olympic
Hang Power Snatch|FB|BB|A|S|ST|C|olympic
Barbell Complex|FB|BB|A|H|WL,EN|C|complex
Dumbbell Thruster|FB|DB|I|H|WL,EN|C|thruster
Dumbbell Snatch|FB|DB|I|H|ST,WL|C|olympic
Man Maker|FB|DB|A|H|WL,ST|C|thruster
Devil Press|FB|DB|A|H|WL,EN|C|thruster
Dumbbell Squat to Press|FB|DB|B|S|GF,WL|C|thruster
Dumbbell Complex|FB|DB|I|H|WL,EN|C|complex
Farmer's Carry|FB|DB|B|S|ST,GF|C|carry
Kettlebell Clean and Press|FB|KB|I|S|ST,GF|C|olympic
Kettlebell Snatch|FB|KB|A|H|EN,WL|C|olympic
Turkish Get-Up|FB|KB|A|S|ST,GF|C|getup
Kettlebell Farmer's Carry|FB|KB|B|S|ST,EN|C|carry
Resistance Band Squat to Row|FB|RB|B|S|GF,EN|C|complex
Sled Push|FB|FG|I|H|WL,ST|C|sled
Sled Pull|FB|FG|I|H|EN,ST|C|sled
Battle Ropes|FB|FG|I|H|WL,EN|C|ropes
Tire Flip|FB|FG|A|S|ST|C|hinge
Medicine Ball Slam|FB|FG|I|H|WL,EN|C|plyo
Wall Ball|FB|FG|I|H|WL,EN|C|thruster
Jumping Jacks|CA|NE|B|C|WL,GF|C|cardio_bw
High Knees|CA|NE|B|H|WL,EN|C|cardio_bw
Butt Kicks|CA|NE|B|C|GF,WL|C|cardio_bw
Jog in Place|CA|NE|B|C|WL,EN|C|cardio_bw
Shadow Boxing|CA|NE|B|C|WL,GF|C|cardio_bw
Lateral Shuffle|CA|NE|B|C|GF,EN|C|cardio_bw
Mountain Climbers|CA|NE|I|H|WL,EN|C|cardio_bw
Skater Hops|CA|NE|I|H|WL,EN|C|plyo
Star Jumps|CA|NE|I|H|WL|C|plyo
Tuck Jumps|CA|NE|A|H|WL,EN|C|plyo
Brisk Walking|CA|NE|B|C|WL,GF|C|cardio_steady
Stair Climbing|CA|NE|B|C|EN,WL|C|cardio_steady
Outdoor Sprint Intervals|CA|NE|A|H|WL,EN|C|cardio_interval
Tabata Bodyweight Circuit|CA|NE|A|H|WL,EN|C|cardio_interval
Incline Treadmill Walk|CA|TM|B|C|WL,GF|C|cardio_steady
Treadmill Jog|CA|TM|B|C|EN,WL|C|cardio_steady
Treadmill Run|CA|TM|I|C|EN,WL|C|cardio_steady
Treadmill Hill Intervals|CA|TM|I|H|WL,EN|C|cardio_interval
Treadmill Sprint Intervals|CA|TM|A|H|WL,EN|C|cardio_interval
Rowing Machine Steady State|CA|MC|B|C|EN,WL|C|cardio_steady
Rowing Machine Intervals|CA|MC|I|H|WL,EN|C|cardio_interval
Stationary Bike|CA|MC|B|C|EN,WL,GF|C|cardio_steady
Spin Bike Intervals|CA|MC|I|H|WL,EN|C|cardio_interval
Elliptical Trainer|CA|MC|B|C|WL,GF|C|cardio_steady
Stair Climber Machine|CA|MC|I|C|EN,WL|C|cardio_steady
Assault Bike Sprints|CA|FG|A|H|WL,EN|C|cardio_interval
Ski Erg Steady State|CA|FG|I|C|EN,WL|C|cardio_steady
Ski Erg Intervals|CA|FG|A|H|WL,EN|C|cardio_interval
Resistance Band Jumping Jacks|CA|RB|I|C|WL,EN|C|cardio_bw
Kettlebell Swing Intervals|CA|KB|A|H|WL,EN|C|cardio_interval
Dumbbell Shadow Boxing|CA|DB|I|C|WL,EN|C|cardio_bw
"""

SETUP = {
    "No Equipment": "Find a clear, non-slip floor space",
    "Dumbbells": "Pick dumbbells you can control for every repetition",
    "Barbell": "Load a barbell with a weight you can control and secure the collars",
    "Resistance Band": "Anchor or hold a resistance band with a suitable tension",
    "Cable": "Set the cable pulley to the right height and choose a manageable stack weight",
    "Machine": "Adjust the seat and pads so the working joint lines up with the machine pivot",
    "Bench": "Use a stable, flat bench or box on a non-slip surface",
    "Kettlebell": "Choose a kettlebell you can control through the full movement",
    "Treadmill": "Clip on the safety key and start the treadmill at a slow walking speed",
    "Pull-up Bar": "Grip a securely mounted pull-up bar",
    "Full Gym": "Set up the station as instructed and check it is stable before starting",
}

# pattern -> (execution cue, safety note, keyword tags)
PATTERNS: dict[str, tuple[str, str, str]] = {
    "h_push": ("Brace your core, lower under control until the elbows are about 45 degrees from the body, then press back to full arm extension.",
               "Keep wrists stacked over elbows and avoid flaring the elbows or arching the lower back excessively.",
               "push press horizontal pectorals triceps"),
    "incline_push": ("On a 30-45 degree incline, lower to the upper chest with elbows slightly tucked, then press up and slightly back.",
                     "Keep shoulder blades retracted and feet planted; use a spotter for heavy sets.",
                     "push press incline upper-chest pectorals deltoids"),
    "fly": ("Keep a soft bend in the elbows, open the arms in a wide arc until a stretch is felt across the chest, then squeeze back together.",
            "Do not overstretch the shoulders at the bottom; use a light weight and a controlled range.",
            "fly pectorals stretch controlled tension"),
    "dip": ("Keep shoulders down and away from the ears, lower until the elbows reach about 90 degrees, then push back to the start.",
            "Stop if you feel pain at the front of the shoulder; avoid dropping too deep.",
            "push dip triceps pectorals"),
    "v_push": ("Brace the core and squeeze the glutes, press overhead until the arms are straight, then lower to shoulder height under control.",
               "Avoid leaning back; keep ribs down and the bar or weights moving in a straight vertical path.",
               "push press overhead vertical deltoids triceps"),
    "lateral_raise": ("With a slight elbow bend, raise the arms to about shoulder height leading with the elbows, pause, then lower slowly.",
                      "Use light weights and avoid shrugging or swinging the torso.",
                      "raise deltoids shoulders controlled tension"),
    "upright_row": ("Pull the weight close to the body to about lower-chest height with the elbows leading, then lower under control.",
                    "Use a moderate grip width and stop below shoulder height to protect the shoulder joint.",
                    "pull upright deltoids traps"),
    "rear_delt": ("Pull the arms out and back by squeezing the shoulder blades together, pause briefly, then return slowly.",
                  "Keep the neck neutral and use light resistance; avoid jerking the weight.",
                  "pull rear-deltoids rhomboids posture"),
    "h_pull": ("Keep a neutral spine, pull toward the lower ribs, squeeze the shoulder blades together, then extend the arms fully.",
               "Avoid rounding the back or using momentum; keep the core braced throughout.",
               "pull row horizontal lats rhomboids biceps"),
    "v_pull": ("Start from a full stretch, pull the elbows down toward the ribs until the chin clears the bar or the handle reaches the upper chest, then return slowly.",
               "Avoid kipping or swinging; control the lowering phase to protect the shoulders and elbows.",
               "pull vertical lats biceps back-width"),
    "hang": ("Hang with straight arms and engaged shoulders, breathing steadily for the prescribed time.",
             "Step down carefully; stop if you feel sharp shoulder or elbow pain.",
             "grip hang lats decompression isometric"),
    "hinge": ("Push the hips back with a neutral spine and soft knees, lower until a hamstring stretch is felt, then drive the hips forward to stand tall.",
              "Never round the lower back; keep the weight close to the legs and start light.",
              "hinge hip-extension hamstrings glutes posterior-chain"),
    "squat": ("Keep the chest up and knees tracking over the toes, sit down to about parallel or a comfortable depth, then stand by pushing through the whole foot.",
              "Keep heels down and knees in line with toes; use safety pins or a spotter for heavy barbell work.",
              "squat quadriceps glutes legs knee-dominant"),
    "lunge": ("Step into a split stance, lower until both knees are near 90 degrees with the front knee over the mid-foot, then drive back up.",
              "Keep the torso upright and use support for balance if needed.",
              "lunge unilateral quadriceps glutes legs balance"),
    "glute_bridge": ("With feet flat and knees bent, drive through the heels to lift the hips until the body forms a straight line from shoulders to knees, squeeze the glutes, then lower.",
                     "Avoid over-arching the lower back at the top; pad the bar for loaded versions.",
                     "bridge hip-extension glutes hips"),
    "abduction": ("Keep the pelvis stable and move the leg out or back using the glutes, pause briefly, then return with control.",
                  "Avoid rotating the torso or swinging the leg; use light resistance.",
                  "abduction glutes hips activation"),
    "calf": ("Rise onto the balls of the feet as high as possible, pause at the top, then lower the heels slowly for a full stretch.",
             "Hold support for balance and avoid bouncing at the bottom.",
             "calves ankle legs"),
    "leg_iso": ("Move only at the knee joint through a full, controlled range, pausing briefly in the contracted position.",
                "Use a weight you can control; avoid locking or snapping the knee.",
                "isolation hamstrings quadriceps knee legs"),
    "curl": ("Keep the elbows close to your sides, curl the weight up by bending the elbows, squeeze the biceps, then lower slowly.",
             "Avoid swinging the torso; keep wrists neutral.",
             "curl biceps arms elbow-flexion isolation"),
    "tri_ext": ("Keep the upper arms still, straighten the elbows fully, squeeze the triceps, then return slowly.",
                "Control the weight near the head and elbows; stop if you feel elbow pain.",
                "extension triceps arms elbow-extension"),
    "shrug": ("Hold the weight at arm's length, lift the shoulders straight up toward the ears, pause, then lower.",
              "Do not roll the shoulders; keep the neck relaxed.",
              "shrug traps upper-back"),
    "back_ext": ("With a neutral neck, lift the torso or legs using the lower back and glutes until the body is in a straight line, then lower slowly.",
                 "Avoid hyperextending past a straight line; move slowly.",
                 "extension lower-back erectors glutes posterior-chain"),
    "plank": ("Hold a straight line from head to heels (or knees), brace the abdominals and glutes, and breathe steadily.",
              "Do not let the hips sag or pike; stop if the lower back aches.",
              "plank isometric core abdominals stability"),
    "iso_hold": ("Hold the position with steady breathing and good alignment for the prescribed time.",
                 "Stop if you feel joint pain; reduce the hold time if form breaks down.",
                 "isometric hold endurance stability"),
    "stability": ("Brace the core so the torso does not move, perform the slow limb or press movement, then return and alternate sides.",
                  "Keep the lower back neutral and move slowly; quality over speed.",
                  "anti-rotation stability core abdominals control"),
    "core_flex": ("Exhale as you curl the ribs toward the pelvis or lift the legs, keep the lower back controlled, then lower slowly without swinging.",
                  "Do not pull on the neck; reduce range if you feel lower-back strain.",
                  "flexion abdominals core crunch"),
    "rotation": ("Brace the core and rotate through the torso in a controlled arc with stable hips, then return through the centre.",
                 "Rotate from the upper back, not the lower back; start without weight.",
                 "rotation obliques core abdominals"),
    "carry": ("Stand tall with shoulders back and walk with short, controlled steps for the prescribed time.",
              "Use a weight you can hold without leaning; set it down carefully.",
              "carry grip core posture functional"),
    "olympic": ("Start with a flat back, extend the hips explosively to accelerate the weight, receive it in a stable position, then return it to the start under control.",
                "Technical lift: learn it with light weight or coaching before loading heavily.",
                "explosive power olympic hip-extension full-body"),
    "thruster": ("Squat with the weight at the shoulders, drive up explosively and use that momentum to press overhead in one movement.",
                 "Keep the core braced and elbows up in the squat; choose a load you can move with speed.",
                 "squat press explosive full-body metabolic"),
    "complex": ("Perform each movement of the sequence back-to-back with smooth transitions and good form before resting.",
                "Pick the weight for your weakest movement in the sequence; rest if form slips.",
                "complex circuit full-body metabolic conditioning"),
    "getup": ("From lying with the weight locked out overhead, move step by step to standing with the arm vertical, then reverse slowly.",
              "Keep your eyes on the weight; practise first without load.",
              "stability shoulder full-body functional control"),
    "plyo": ("Land softly with bent knees, stay light on your feet and move explosively while keeping good form for the prescribed time.",
             "Use a forgiving surface, warm up first and reduce impact if you have joint concerns.",
             "plyometric explosive jump metabolic conditioning calorie-burn"),
    "crawl": ("On hands and feet with knees just off the floor, move opposite hand and foot together while keeping the back flat.",
              "Keep wrists under shoulders and move slowly at first.",
              "crawl coordination core full-body conditioning"),
    "sled": ("Lean into the sled with a straight body line and drive with short, powerful steps for the prescribed time.",
             "Keep the back flat and choose a load you can move continuously.",
             "sled drive legs full-body conditioning power"),
    "ropes": ("In a half-squat stance, create continuous waves with both ropes while keeping the core braced.",
              "Keep the back neutral and shoulders relaxed; stop if form breaks down.",
              "ropes interval metabolic conditioning shoulders calorie-burn"),
    "cardio_steady": ("Start easy for a minute, hold a steady conversational effort for the prescribed time, then finish with a slower minute.",
                      "Increase pace gradually and stay hydrated; stop if you feel dizzy or unwell.",
                      "aerobic steady cardio heart-rate stamina sustained"),
    "cardio_interval": ("Alternate hard efforts for the prescribed work time with easy recovery during the rest period.",
                        "Warm up thoroughly first; high-intensity intervals are demanding on the heart and joints.",
                        "interval sprint metabolic conditioning cardio calorie-burn"),
    "cardio_bw": ("Move continuously at a brisk, rhythmic pace for the prescribed time, landing softly and breathing steadily.",
                  "Wear supportive footwear and reduce the pace if needed.",
                  "aerobic cardio rhythmic conditioning heart-rate calorie-burn"),
    "mobility": ("Move slowly into the position, breathe deeply, and hold or repeat gently without forcing the range.",
                 "Stretch to mild tension only, never to pain; avoid bouncing.",
                 "mobility stretch flexibility range-of-motion recovery"),
}

TIMED_PATTERNS = {"hang", "plank", "iso_hold", "carry", "plyo", "crawl", "sled",
                  "ropes", "cardio_steady", "cardio_interval", "cardio_bw", "mobility"}

EQUIP_CLASS = {
    "No Equipment": "bodyweight", "Dumbbells": "free-weight dumbbell",
    "Barbell": "free-weight barbell heavy", "Resistance Band": "elastic band",
    "Cable": "cable constant-tension", "Machine": "machine guided",
    "Bench": "bodyweight bench", "Kettlebell": "free-weight kettlebell",
    "Treadmill": "cardio-machine", "Pull-up Bar": "bodyweight bar",
    "Full Gym": "gym-station",
}
TYPE_TAGS = {"Strength": "resistance strength", "Cardio": "aerobic cardio",
             "HIIT": "interval explosive metabolic", "Mobility": "mobility flexibility",
             "Core": "core stability"}

# Approximate MET values by type and difficulty (educational estimates)
MET = {
    "Strength": {"Beginner": 3.5, "Intermediate": 5.0, "Advanced": 6.0},
    "Core": {"Beginner": 3.0, "Intermediate": 3.8, "Advanced": 4.5},
    "HIIT": {"Beginner": 7.0, "Intermediate": 8.5, "Advanced": 10.0},
    "Cardio": {"Beginner": 4.5, "Intermediate": 7.0, "Advanced": 9.0},
    "Mobility": {"Beginner": 2.3, "Intermediate": 2.5, "Advanced": 2.8},
}


def base_prescription(goal: str, ex_type: str, pattern: str, mechanics: str,
                      difficulty: str) -> tuple[int, str, int]:
    """Dataset prescription (sets, reps, rest) for an exercise's PRIMARY goal."""
    d = difficulty
    if ex_type == "Mobility" or pattern == "mobility":
        return 2, "30 sec", 15
    if pattern == "cardio_steady":
        return 1, {"Beginner": "10 min", "Intermediate": "15 min", "Advanced": "20 min"}[d], 0
    if pattern == "cardio_interval":
        return ({"Beginner": 5, "Intermediate": 6, "Advanced": 8}[d],
                {"Beginner": "20 sec", "Intermediate": "30 sec", "Advanced": "30 sec"}[d],
                {"Beginner": 60, "Intermediate": 60, "Advanced": 45}[d])
    if ex_type in ("HIIT", "Cardio") or pattern in {"plyo", "crawl", "sled", "ropes", "cardio_bw"}:
        return ({"Beginner": 3, "Intermediate": 4, "Advanced": 4}[d],
                {"Beginner": "30 sec", "Intermediate": "40 sec", "Advanced": "45 sec"}[d],
                {"Beginner": 45, "Intermediate": 30, "Advanced": 30}[d])
    if pattern in {"plank", "iso_hold", "hang"}:
        return 3, {"Beginner": "30 sec", "Intermediate": "45 sec", "Advanced": "60 sec"}[d], 45
    if pattern == "carry":
        return 3, "40 sec", 60
    if ex_type == "Core":
        reps = {"Beginner": "10-12", "Intermediate": "12-15", "Advanced": "12-15"}[d]
        if pattern == "stability":
            reps = "8-10"
        return 3, reps, 45
    sets, reps, rest = GOAL_PRESCRIPTION[goal]["compound" if mechanics == "Compound" else "isolation"]
    if d == "Beginner":
        sets = max(2, sets - 1)
    return sets, reps, rest


def build_dataframe() -> pd.DataFrame:
    rows = []
    for line in CATALOGUE.strip().splitlines():
        name, m, e, d, t, g, mech, pattern = [x.strip() for x in line.split("|")]
        muscle, equip, diff, ex_type = MUSCLE[m], EQUIP[e], DIFF[d], TYPE[t]
        goals = [GOAL[x] for x in g.split(",")]
        mechanics = MECH[mech]
        cue, safety, ptags = PATTERNS[pattern]
        sets, reps, rest = base_prescription(goals[0], ex_type, pattern, mechanics, diff)
        seconds = estimate_exercise_seconds(sets, reps, rest)
        minutes = round(seconds / 60.0, 1)
        kcal_per_min = MET[ex_type][diff] * 3.5 * REFERENCE_BODY_WEIGHT_KG / 200.0
        calories = int(round(kcal_per_min * minutes))
        instructions = (f"1) Setup: {SETUP[equip]}. 2) Execute: {cue} "
                        f"3) Complete the prescribed {('time' if pattern in TIMED_PATTERNS else 'repetitions')} "
                        f"with steady breathing.")
        diff_tag = {"Beginner": "beginner-friendly", "Intermediate": "moderate",
                    "Advanced": "challenging"}[diff]
        tags = " ".join([
            name.lower().replace("'", "").replace("/", " "),
            ptags, mechanics.lower(), EQUIP_CLASS[equip], TYPE_TAGS[ex_type],
            muscle.lower().replace(" ", "-"), diff_tag,
        ])
        rows.append({
            "exercise_name": name, "muscle_group": muscle, "equipment": equip,
            "difficulty": diff, "goal": "|".join(goals), "exercise_type": ex_type,
            "duration_minutes": minutes, "calories_burned": calories,
            "sets": sets, "reps": reps, "rest_seconds": rest,
            "instructions": instructions, "safety_notes": safety,
            "mechanics": mechanics, "movement_pattern": pattern, "tags": tags,
        })
    df = pd.DataFrame(rows)
    if df["exercise_name"].duplicated().any():
        dups = df.loc[df["exercise_name"].duplicated(), "exercise_name"].tolist()
        raise ValueError(f"Duplicate exercise names in catalogue: {dups}")
    df.insert(0, "exercise_id", [f"EX{i:03d}" for i in range(1, len(df) + 1)])
    return df


def main() -> None:
    df = build_dataframe()
    out = ROOT / "data" / "exercises.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} exercises to {out}")


if __name__ == "__main__":
    main()
