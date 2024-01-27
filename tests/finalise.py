import os

# Delete olsr file


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


if __name__ == "main":
    delete_file("test.csv")
