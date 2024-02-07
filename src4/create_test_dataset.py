import os
import shutil
import random

def move_files(source_folder, target_folder, fraction=0.1):
    # Pobranie listy plików w folderze źródłowym
    files = os.listdir(source_folder)

    # Wybór losowego podzbioru plików do przeniesienia
    num_files_to_move = int(fraction * len(files))
    files_to_move = random.sample(files, num_files_to_move)

    # Utworzenie folderu docelowego, jeśli jeszcze nie istnieje
    os.makedirs(target_folder, exist_ok=True)

    # Przeniesienie wybranych plików do folderu docelowego
    for file in files_to_move:
        shutil.move(os.path.join(source_folder, file), target_folder)

# Użycie funkcji
source_folder = '..//data//parts6s_resampled'
target_folder = '..//data//parts6s_resampled_test'
move_files(source_folder, target_folder, fraction=0.1)