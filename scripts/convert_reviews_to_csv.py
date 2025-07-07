#!/usr/bin/env python

import json
import csv
import re
from collections import defaultdict

def extract_dandiset_id(notebook_uri):
    # Extract dandiset ID from URIs like:
    # https://github.com/dandi-ai-notebooks/dandi-ai-notebooks-6/blob/main/notebooks/dandisets/001433/0.250507.2356/6e10365d/claude-sonnet-4/h-5/notebook.ipynb
    match = re.search(r'/dandisets/(\d+)/', notebook_uri)
    if match:
        # Ensure dandiset ID is returned as a string with leading zeros
        return match.group(1).zfill(6)
    return None

def main():
    # Read the JSON file
    with open('reviews-export-2025-07-02.json', 'r') as f:
        reviews = json.load(f)

    # Track all unique question IDs to create CSV headers
    question_ids = set()

    # Filter and process completed reviews (excluding specific reviewer)
    processed_reviews = []
    for review in reviews:
        if (review['review']['status'] == 'completed' and
            review['reviewer_email'] != 'jmagland@flatironinstitute.org'):
            # Extract responses
            responses_dict = {}
            for response in review['review']['responses']:
                question_ids.add(response['question_id'])
                responses_dict[response['question_id']] = response['response']

            # Extract dandiset ID
            dandiset_id = extract_dandiset_id(review['notebook_uri'])

            # Store processed review
            processed_reviews.append({
                'reviewer_email': review['reviewer_email'],
                'dandiset_id': dandiset_id,
                **responses_dict
            })

    # Create CSV headers
    headers = ['reviewer_email', 'dandiset_id'] + sorted(list(question_ids))

    # Write to CSV
    output_file = 'reviews_export.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for review in processed_reviews:
            writer.writerow(review)

    print(f'CSV file has been created: {output_file}')
    print(f'Total completed reviews processed: {len(processed_reviews)}')

if __name__ == '__main__':
    main()
