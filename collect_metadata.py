import pandas
import requests


def collect_metadata(filepath, filename):
    print(f"Collecting metadata for {filename}")
    dataframe = pandas.read_csv(filepath + filename)

    image_ids = []
    image_type = []
    sex = []
    age_approx = []
    anatom_site_general = []
    diagnosis_confirm_type = []
    missing_age_counter = 0
    missing_sex_counter = 0

    for image_id in dataframe["image"]:
        if int(image_id.split("_")[1]) % 100 == 0:
            print(image_id)
        image_ids.append(image_id)
        response = requests.get(url=f"https://api.isic-archive.com/api/v2/images/{image_id}/")
        meta_data = response.json()['metadata']
        if 'image_type' in meta_data['acquisition']:
            image_type.append(meta_data['acquisition']['image_type'])
        else:
            image_type.append("unknown")
        if 'sex' in meta_data['clinical']:
            sex.append(meta_data['clinical']['sex'])
        else:
            sex.append("")
            missing_sex_counter = missing_sex_counter + 1
        if 'age_approx' in meta_data['clinical']:
            age_approx.append(meta_data['clinical']['age_approx'])
        else:
            age_approx.append("")
            missing_age_counter = missing_age_counter + 1
        if 'anatom_site_general' in meta_data['clinical']:
            anatom_site_general.append(meta_data['clinical']['anatom_site_general'])
        else:
            anatom_site_general.append("unknown")
        if 'diagnosis_confirm_type' in meta_data['clinical']:
            diagnosis_confirm_type.append(meta_data['clinical']['diagnosis_confirm_type'])
        else:
            diagnosis_confirm_type.append("unknown")

    meta_data = {"image": image_ids, "image_type": image_type, "sex": sex, "age_approx": age_approx, "anatom_site_general": anatom_site_general, "diagnosis_confirm_type": diagnosis_confirm_type}

    dataframe = pandas.DataFrame(meta_data)
    dataframe.to_csv(filepath + "metadata.csv", index=False)
    print(f"Missing age: {missing_age_counter}")
    print(f"Missing sex: {missing_sex_counter}")


if __name__ == "__main__":
    collect_metadata("data/ISIC2018_Task3_Test_GroundTruth/", "ISIC2018_Task3_Test_GroundTruth.csv")
    collect_metadata("data/ISIC2018_Task3_Validation_GroundTruth/", "ISIC2018_Task3_Validation_GroundTruth.csv")
    collect_metadata("data/ISIC2018_Task3_Training_GroundTruth/", "ISIC2018_Task3_Training_GroundTruth.csv")
