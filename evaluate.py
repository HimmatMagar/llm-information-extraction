from main import extract_information


TEST_CASES = [
    {
        "text": """
        Alice is a Data Scientist at Amazon in Seattle.
        She has 4 years of experience.
        Email: alice@example.com
        """,
        "expected": {
            "name": "Alice",
            "job_title": "Data Scientist",
            "company": "Amazon",
            "location": "Seattle",
            "years_experience": 4,
            "email": "alice@example.com",
        },
    },

    {
        "text": """
        Bob works as a Software Engineer at Microsoft in London.
        He has 5 years of experience.
        Contact: bob@example.com
        """,
        "expected": {
            "name": "Bob",
            "job_title": "Software Engineer",
            "company": "Microsoft",
            "location": "London",
            "years_experience": 5,
            "email": "bob@example.com",
        },
    },

    {
        "text": """
        Carol is a Machine Learning Engineer at Google in California.
        She has 3 years of experience.
        Email: carol@example.com
        """,
        "expected": {
            "name": "Carol",
            "job_title": "Machine Learning Engineer",
            "company": "Google",
            "location": "California",
            "years_experience": 3,
            "email": "carol@example.com",
        },
    },
]


def normalize(value):

    if isinstance(value, str):
        return value.strip().lower()

    return value


correct_examples = 0
correct_fields = 0
total_fields = 0


for i, case in enumerate(TEST_CASES, start=1):

    try:

        result = extract_information(case["text"])

        predicted = result.model_dump()
        expected = case["expected"]

        field_results = {}

        for field in expected:

            is_correct = (
                normalize(predicted[field])
                == normalize(expected[field])
            )

            field_results[field] = is_correct

        example_correct = all(field_results.values())

        if example_correct:
            correct_examples += 1

        correct_fields += sum(field_results.values())
        total_fields += len(field_results)

        print(
            f"Example {i}: "
            f"{'PASS' if example_correct else 'FAIL'}"
        )

        for field, correct in field_results.items():

            print(
                f"  {field}: "
                f"{'OK' if correct else 'WRONG'}"
            )

    except Exception as e:

        print(f"Example {i}: ERROR")
        print(e)

        total_fields += len(case["expected"])


print("\n========== RESULTS ==========")

example_accuracy = (
    correct_examples / len(TEST_CASES) * 100
)

field_accuracy = (
    correct_fields / total_fields * 100
)

print(
    f"Example accuracy: "
    f"{example_accuracy:.2f}%"
)

print(
    f"Field accuracy: "
    f"{field_accuracy:.2f}%"
)