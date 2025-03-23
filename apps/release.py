import os
import glob


def delete_mp4_files(directory="."):
    mp4_files = glob.glob(os.path.join(directory, '*.mp4'))

    for file_path in mp4_files:
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 时出错: {e}")

if __name__ == '__main__':
    delete_mp4_files()
