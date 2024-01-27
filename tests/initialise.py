import csv


# Generate test CSV to test olsr regressor
def generate_test_csv():
    csv_path = "test.csv"
    column_names = ["feature1", "feature2", "target"]
    data = [[1, 2, 7], [3, 7, 8], [2, 6, 9]]

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(data)


if __name__ == "__main__":
    generate_test_csv()
