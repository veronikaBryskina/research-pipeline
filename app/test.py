from src.data_preprocessing import DataProcessor

file = DataProcessor()
#file.upload_data("data/01_input/Fake.csv")
content = file.fetch_data("Fake.csv")

print(content.head())