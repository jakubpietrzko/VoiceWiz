import os
import shutil
from glob import glob

# Ścieżki do folderów źródłowych
source_dirs = ['..//data//fzeros', '..//data//mels', '..//data//parts6s']
first= True
# Ścieżki do folderów docelowych
target_dirs = ['..//data//f', '..//data//y', '..//data//x']
names = set()
# Liczba plików do skopiowania
num_files_to_copy = 32

for source_dir, target_dir in zip(source_dirs, target_dirs):
    # Pobierz listę plików w folderze źródłowym
    files = glob(os.path.join(source_dir, '*'))

    # Upewnij się, że jest wystarczająco dużo plików do skopiowania
    if len(files) < num_files_to_copy:
        print(f"Folder {source_dir} zawiera mniej niż {num_files_to_copy} plików.")
        continue

    # Skopiuj pierwsze 'num_files_to_copy' plików
    if first:
        for file in files[:num_files_to_copy]:
            shutil.copy(file, target_dir)
            filename, _ = os.path.splitext(os.path.basename(file))
            names.add(filename)
    else:
        for file in files:
            filename, _ = os.path.splitext(os.path.basename(file))
            if filename in names:
                shutil.copy(file, target_dir)
    first = False