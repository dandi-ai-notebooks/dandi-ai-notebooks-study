#!/usr/bin/env python

import json
import csv
import re
from collections import defaultdict

def get_reviewer_id(email, email_to_id=None):
    """Map reviewer email to anonymous ID (R1, R2, etc)"""
    if email_to_id is None:
        email_to_id = {}
    if email not in email_to_id:
        reviewer_num = len(email_to_id) + 1
        email_to_id[email] = f'R{reviewer_num}'
        # email_to_id[email] = email
    return email_to_id[email]

def extract_dandiset_id(notebook_uri):
    # Extract dandiset ID from URIs like:
    # https://github.com/dandi-ai-notebooks/dandi-ai-notebooks-6/blob/main/notebooks/dandisets/001433/0.250507.2356/6e10365d/claude-sonnet-4/h-5/notebook.ipynb
    match = re.search(r'/dandisets/(\d+)/', notebook_uri)
    if match:
        # Ensure dandiset ID is returned as a string with leading zeros
        return match.group(1).zfill(6)
    return None

def _is_author_email(email):
    x = ['jmagland@', 'oruebel@', 'rly@', 'ben.dichter@']
    # Check if the email belongs to the author
    return any(email.startswith(prefix) for prefix in x)

def main():
    # Read the JSON file
    with open('reviews-export-2025-07-14.json', 'r') as f:
        reviews = json.load(f)

    # Track all unique question IDs to create CSV headers
    question_ids = set()
    email_to_id = {}  # Map to track email->anonymous ID mapping

    # Filter and process completed reviews (excluding specific reviewer)
    processed_reviews = []
    for review in reviews:
        if (review['review']['status'] == 'completed' and
            not _is_author_email(review['reviewer_email'])):
            # Extract responses and rationales
            responses_dict = {}
            for response in review['review']['responses']:
                question_id = response['question_id']
                question_ids.add(question_id)
                responses_dict[question_id] = response['response']
                responses_dict[f"{question_id}-rationale"] = response.get('rationale', '')

            # Extract dandiset ID
            dandiset_id = extract_dandiset_id(review['notebook_uri'])

            # Store processed review with anonymous reviewer ID
            processed_reviews.append({
                'reviewer_id': get_reviewer_id(review['reviewer_email'], email_to_id),
                'dandiset_id': dandiset_id,
                **responses_dict
            })

    # Create CSV headers including rationale columns
    question_headers = []
    for qid in sorted(list(question_ids)):
        question_headers.append(qid)
        question_headers.append(f"{qid}-rationale")
    headers = ['reviewer_id', 'dandiset_id'] + question_headers

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
