import csv
import json
import uuid
import boto3
from tqdm import tqdm
import time
import random

# Initialize a Kinesis client
kinesis_client = boto3.client("kinesis", region_name="us-east-1")


from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def ingest(data):
    try:
        response = kinesis_client.put_record(
            StreamName="KinesisStream",
            Data=json.dumps(data),
            PartitionKey=data["uid"],
        )
        return response
    except Exception as e:
        print(f"Error ingesting data: {e}")
        raise


def main(file_path):
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file, delimiter="\t")
        for row in tqdm(csv_reader):
            # Simulate network latency
            time.sleep(random.uniform(0, 1))

            # Ingest display event
            display_data = {
                "event_type": "display",
                "timestamp": row["timestamp"],
                "uid": row["uid"],
                "campaign": row["campaign"],
                "cost": row["cost"],
                "attribution": row["attribution"],
                "cat1": row["cat1"],
                "cat2": row["cat2"],
                "cat3": row["cat3"],
                "cat4": row["cat4"],
                "cat5": row["cat5"],
                "cat6": row["cat6"],
                "cat7": row["cat7"],
                "cat8": row["cat8"],
                "cat9": row["cat9"],
            }
            ingest(display_data)

            # Ingest click event if applicable
            if row["click"] == "1":
                click_data = {
                    "event_type": "click",
                    "timestamp": row["timestamp"],
                    "uid": row["uid"],
                    "campaign": row["campaign"],
                }
                ingest(click_data)

            # Ingest conversion event if applicable
            if row["conversion"] == "1":
                conversion_data = {
                    "event_type": "conversion",
                    "timestamp": row["timestamp"],
                    "uid": row["uid"],
                    "campaign": row["campaign"],
                    "conversion_timestamp": row["conversion_timestamp"],
                    "conversion_id": row["conversion_id"],
                    "cpo": row["cpo"],
                }
                ingest(conversion_data)


# Example usage
if __name__ == "__main__":
    main("data/criteo_attribution_dataset.tsv")
