import os
import glob


def delete_v_files(directory="."):
    webm_files = glob.glob(os.path.join(directory, '*.webm'))

    for file_path in webm_files:
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 时出错: {e}")
def delete_zip_files(directory="."):
    zip_files = glob.glob(os.path.join(directory, '*.zip'))

    for file_path in zip_files:
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 时出错: {e}")

def delete_temp_files():
    delete_v_files()
    delete_zip_files()

if __name__ == '__main__':
    delete_temp_files()
