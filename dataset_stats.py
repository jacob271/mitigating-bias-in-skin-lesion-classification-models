import pandas
from matplotlib import pyplot as plt

df_images = pandas.read_csv("data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

print(df_images.head())
print("-------------------------")
print(f"Number of images: {len(df_images.index)}")
for column in df_images:
    if column == "image":
        continue
    print(f"Number of {column} images: {len(df_images[df_images[column] == 1].index)}")


df_metadata = pandas.read_csv("data/HAM10000_metadata.csv")
print(f"Unique age values: {df_metadata['age'].unique()}")
print(f"Unique values dx_type: {df_metadata['dx_type'].unique()}")

for column in df_images:
    stats = {"male": 0, "female": 0,
             "age": {"85.0": 0, "80.0": 0, "75.0": 0, "70.0": 0, "65.0": 0, "60.0": 0, "55.0": 0, "50.0": 0, "45.0": 0, "40.0": 0, "35.0": 0, "30.0": 0, "25.0": 0, "20.0": 0, "15.0": 0, "10.0": 0, "5.0": 0, "0.0": 0, "nan": 0},
             "dx_type": {"histo": 0, "consensus": 0, "confocal": 0, "follow_up": 0}
             }
    if column == "image":
        continue
    for image_id in df_images[df_images[column] == 1]["image"]:
        if df_metadata[df_metadata["image_id"] == image_id]["sex"].iloc[0] == "male":
            stats["male"] = stats["male"] + 1
        else:
            stats["female"] = stats["female"] + 1

        age = df_metadata[df_metadata["image_id"] == image_id]["age"]

        stats["age"][str(age.iloc[0])] = stats["age"][str(age.iloc[0])] + 1

        dx_type = df_metadata[df_metadata["image_id"] == image_id]["dx_type"].iloc[0]
        stats["dx_type"][dx_type] = stats["dx_type"][dx_type] + 1
    plt.title(f"Age distribution for class {column}")
    plt.xlabel("Age")
    plt.ylabel("Number of images")
    plt.bar(stats["age"].keys(), stats["age"].values(), width=1.0)
    plt.xticks(rotation=45, ha='right')
    plt.show()
    print(f"Stats for {column}:")
    stats.pop("age")
    print(stats)
